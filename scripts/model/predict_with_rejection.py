"""
Roda um modelo ja treinado e aplica um limiar de
confianca: previsoes cuja confianca fique abaixo do limiar sao
reclassificadas como "outros" (classe desconhecida), em vez de forcar
uma das classes vistas no treino.

Uso 
    python3 ./scripts/model/predict_with_rejection.py \
        --model ./runs/classify/results/dataset_8c/yolo26l_melhor_opt/yolo26l-cls.pt_dataset_8c_20ep/train/weights/best.pt \
        --source ./cls/dataset_inteiro/val \
        --threshold 0.97 \
        --imgsz 640 \
        --csv-out ./results/predicoes_openset_6.csv

"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
UNKNOWN_LABEL = 'outros'


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classificacao com rejeicao por limiar de confianca"
    )
    p.add_argument('--model', type=str, required=True,
                   help='caminho pro checkpoint treinado (.pt)')
    p.add_argument('--source', type=str, required=True,
                   help='pasta com imagens (soltas ou em subpastas por classe)')
    p.add_argument('--threshold', type=float, default=0.6,
                   help=f'confianca minima pra aceitar a previsao '
                        f'(abaixo disso vira "{UNKNOWN_LABEL}")')
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--csv-out', type=str, required=True)
    return p.parse_args()


def find_images(source: Path, known_names: set):
    entries = list(source.iterdir())
    subdirs = [e for e in entries if e.is_dir()]

    images = []
    if subdirs:
        for class_dir in subdirs:
            true_label = class_dir.name
            is_known = true_label in known_names
            for img_path in class_dir.rglob('*'):
                if img_path.suffix.lower() in IMG_EXTS:
                    images.append((img_path, true_label, is_known))
    else:
        for img_path in source.rglob('*'):
            if img_path.suffix.lower() in IMG_EXTS:
                images.append((img_path, None, None))
    return images


def main():
    args = get_args()
    source = Path(args.source)

    model = YOLO(args.model, task='classify')
    names = model.names  
    known_names = set(names.values())

    images = find_images(source, known_names)

    if not images:
        raise SystemExit(f"Nenhuma imagem encontrada em {source}")

    labeled = images[0][1] is not None
    if labeled:
        n_known_species = len({t for _, t, k in images if k})
        n_novel_species = len({t for _, t, k in images if not k})
        print(f"[predict] {len(images)} imagens encontradas "
              f"({n_known_species} especies conhecidas, "
              f"{n_novel_species} especies novas)")
        if n_novel_species == 0:
            print("[predict] aviso: nenhuma subpasta ficou fora de model.names")
    else:
        print(f"[predict] {len(images)} imagens encontradas (sem rotulo)")

    rows = []
    paths = [str(p) for p, _, _ in images]

    for start in range(0, len(paths), args.batch):
        batch_paths = paths[start:start + args.batch]
        results = model.predict(
            source=batch_paths,
            imgsz=args.imgsz,
            verbose=False,
        )
        batch_images = images[start:start + args.batch]
        for (img_path, true_label, is_known), result in zip(batch_images, results):
            probs = result.probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            pred_class = names[top1_idx]

            final_label = pred_class if top1_conf >= args.threshold else UNKNOWN_LABEL

            rows.append({
                'image': str(img_path),
                'true_label': true_label if true_label is not None else '',
                'species_type': ('conhecida' if is_known else 'nova') if true_label else '',
                'predicted_class': pred_class,
                'confidence': round(top1_conf, 5),
                'final_label': final_label,
            })

    os.makedirs(os.path.dirname(args.csv_out) or '.', exist_ok=True)
    header = ['image', 'true_label', 'species_type', 'predicted_class', 'confidence', 'final_label']
    with open(args.csv_out, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(row[h]).replace(',', ';') for h in header) + '\n')

    print(f"[predict] resultados salvos em: {args.csv_out}")

    n_rejected = sum(1 for r in rows if r['final_label'] == UNKNOWN_LABEL)
    print(f"[predict] {n_rejected}/{len(rows)} imagens classificadas como "
          f"'{UNKNOWN_LABEL}' (confianca < {args.threshold})")

    if not labeled:
        return

    known_rows = [r for r in rows if r['species_type'] == 'conhecida']
    novel_rows = [r for r in rows if r['species_type'] == 'nova']

    if known_rows:
        n_known = len(known_rows)
        n_known_rejected = sum(1 for r in known_rows if r['final_label'] == UNKNOWN_LABEL)
        n_correct_accepted = sum(1 for r in known_rows if r['final_label'] == r['true_label'])
        n_accepted = n_known - n_known_rejected

        print(f"\n[predict] especies CONHECIDAS ({n_known} imagens):")
        print(f"    {n_correct_accepted} classificadas corretamente")
        print(f"    {n_known_rejected} rejeitadas como '{UNKNOWN_LABEL}' "
              f"(falso-rejeitado -- eram de especie conhecida)")
        if n_accepted > 0:
            print(f"    accuracy entre as aceitas (nao rejeitadas): "
                  f"{n_correct_accepted / n_accepted:.4f}")

    if novel_rows:
        n_novel = len(novel_rows)
        n_correctly_rejected = sum(1 for r in novel_rows if r['final_label'] == UNKNOWN_LABEL)
        n_wrongly_accepted = n_novel - n_correctly_rejected

        print(f"\n[predict] especies NOVAS / fora do treino ({n_novel} imagens):")
        print(f"    {n_correctly_rejected} corretamente rejeitadas como '{UNKNOWN_LABEL}'")
        print(f"    {n_wrongly_accepted} falso-aceitas como alguma especie conhecida "
              f"(caso mais critico)")
        print(f"    taxa de rejeicao correta: {n_correctly_rejected / n_novel:.4f}")

        by_species = {}
        for r in novel_rows:
            by_species.setdefault(r['true_label'], {'total': 0, 'aceita': 0})
            by_species[r['true_label']]['total'] += 1
            if r['final_label'] != UNKNOWN_LABEL:
                by_species[r['true_label']]['aceita'] += 1
        print(f"    por especie nova (imagens falso-aceitas / total):")
        for species, counts in sorted(by_species.items()):
            print(f"        {species}: {counts['aceita']}/{counts['total']}")


if __name__ == '__main__':
    main()