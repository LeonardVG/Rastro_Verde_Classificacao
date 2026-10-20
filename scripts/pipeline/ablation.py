"""
Ablação individual de augmentations.

Roda um baseline (NoAug) + cada augmentation ligado sozinho (resto zerado),
mesmo número de épocas pra todos, e compara o macro-F1 contra o baseline.

Uso:
    cd scripts/
    python ../pipeline/ablation.py \
        --model yolov8n-cls.pt \
        --epoch 20 \
        --train-path ../cls/dataset/train \
        --val-path ../cls/dataset/val \
        --models-path ../Models/Base \
        --save-models-path ../Models/Ablation \
        --result-path ../Results/ablation \
        --csv-out ../Results/ablation/summary.csv
"""
import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.train import train
from model.bench import bench

AUGMENTATIONS = {
    'fliplr': 0.5,
    'flipud': 0.3,
    'hsv_h': 0.015,
    'hsv_s': 0.4,
    'hsv_v': 0.4,
    'scale': 0.5,
    'erasing': 0.4,
}

ALL_AUG_KEYS = list(AUGMENTATIONS.keys())

AUTO_AUGMENT_PRESETS = ['randaugment', 'autoaugment', 'augmix']


def zeroed_kwargs() -> dict:
    kwargs = {k: 0 for k in ALL_AUG_KEYS}
    kwargs['auto_augment'] = None 
                                    
    return kwargs


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablação individual de augmentations")
    p.add_argument('--model', type=str, default='yolov8n-cls.pt')
    p.add_argument('--epoch', type=int, default=20)
    p.add_argument('--save-period', type=int, default=-1)
    p.add_argument('--train-path', type=str, required=True)
    p.add_argument('--val-path', type=str, required=True)
    p.add_argument('--models-path', type=str, required=True, help='onde estão os pesos pré-treinados (.pt base)')
    p.add_argument('--save-models-path', type=str, required=True)
    p.add_argument('--result-path', type=str, required=True)
    p.add_argument('--csv-out', type=str, required=True)
    p.add_argument('--keep-checkpoints', action='store_true',
                    help='se não passar, apaga o .pt de cada run que não bater o baseline (economiza disco)')
    p.add_argument('--smoke-test', action='store_true',
                    help='roda só baseline + 1 augmentation, com poucas épocas, pra validar o pipeline antes da rodada completa')
    p.add_argument('--only', type=str, default=None,
                    help='lista separada por vírgula de augmentations pra rodar (ex: hsv_h,hsv_s,hsv_v). Se omitido, roda todas.')
    p.add_argument('--imgsz', type=int, default=224,
                    help='resolução da imagem de entrada (ex: 224, 320, 512)')
    p.add_argument('--batch', type=int, default=16,
                    help='batch size (ajuste conforme a VRAM disponível, especialmente em resoluções maiores)')
    return p.parse_args()


def run_one(dataset_name: str, model: str, epoch: int, save_period: int,
            models_path: str, save_models_path: str, result_path: str,
            train_path: str, val_path: str, aug_kwargs: dict,
            imgsz: int, batch: int) -> dict:
    print(f"\n{'='*60}\n[ablation] rodando: {dataset_name}\n{'='*60}")

    best_pt = train(model, epoch, save_period, models_path, save_models_path, result_path,
                     dataset_name, train_path, val_path, imgsz=imgsz, batch=batch, **aug_kwargs)

    metrics = bench(best_pt, epoch, save_period, models_path, save_models_path, result_path, val_path,
                     imgsz=imgsz)
    metrics['best_pt'] = best_pt
    return metrics


def main():
    args = get_args()
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    augmentations_to_run = AUGMENTATIONS
    if args.only:
        wanted = [a.strip() for a in args.only.split(',')]
        augmentations_to_run = {k: v for k, v in AUGMENTATIONS.items() if k in wanted}
        missing = set(wanted) - set(augmentations_to_run.keys())
        if missing:
            print(f"[ablation] aviso: augmentation(s) não encontrada(s), ignorando: {missing}")

    if args.smoke_test:
        args.epoch = min(args.epoch, 2)
        augmentations_to_run = dict(list(augmentations_to_run.items())[:1])
        print(f"[ablation] SMOKE TEST: epoch={args.epoch}, só testando '{list(augmentations_to_run.keys())[0]}'\n")

    rows = []

    baseline_metrics = run_one(
        "NoAug", args.model, args.epoch, args.save_period,
        args.models_path, args.save_models_path, args.result_path,
        args.train_path, args.val_path, zeroed_kwargs(),
        imgsz=args.imgsz, batch=args.batch
    )
    baseline_f1 = baseline_metrics['macro_f1']
    rows.append({'aug': 'baseline_noaug', 'value': 0, 'macro_f1': baseline_f1,
                 'top1': baseline_metrics['top1'], 'top5': baseline_metrics['top5'],
                 'delta_f1': 0.0})
    print(f"\n[ablation] Baseline macro-F1 ({args.epoch}ep): {baseline_f1:.5f}\n")

    for aug_name, aug_value in augmentations_to_run.items():
        kwargs = zeroed_kwargs()
        kwargs[aug_name] = aug_value

        metrics = run_one(
            f"aug-{aug_name}", args.model, args.epoch, args.save_period,
            args.models_path, args.save_models_path, args.result_path,
            args.train_path, args.val_path, kwargs,
            imgsz=args.imgsz, batch=args.batch
        )

        delta = metrics['macro_f1'] - baseline_f1
        rows.append({'aug': aug_name, 'value': aug_value, 'macro_f1': metrics['macro_f1'],
                     'top1': metrics['top1'], 'top5': metrics['top5'], 'delta_f1': delta})

        print(f"[ablation] {aug_name}: macro-F1={metrics['macro_f1']:.5f} (delta={delta:+.5f})")

        if not args.keep_checkpoints and delta <= 0:
            pt_path = os.path.join(args.save_models_path, f"{args.model}_aug-{aug_name}_{args.epoch}ep.pt")
            if os.path.exists(pt_path):
                os.remove(pt_path)

    presets_to_run = AUTO_AUGMENT_PRESETS if not args.smoke_test else []
    for preset in presets_to_run:
        kwargs = zeroed_kwargs()
        kwargs['auto_augment'] = preset

        metrics = run_one(
            f"autoaug-{preset}", args.model, args.epoch, args.save_period,
            args.models_path, args.save_models_path, args.result_path,
            args.train_path, args.val_path, kwargs,
            imgsz=args.imgsz, batch=args.batch
        )

        delta = metrics['macro_f1'] - baseline_f1
        rows.append({'aug': f'auto_augment_{preset}', 'value': preset, 'macro_f1': metrics['macro_f1'],
                     'top1': metrics['top1'], 'top5': metrics['top5'], 'delta_f1': delta})

        print(f"[ablation] auto_augment={preset}: macro-F1={metrics['macro_f1']:.5f} (delta={delta:+.5f})")

        if not args.keep_checkpoints and delta <= 0:
            pt_path = os.path.join(args.save_models_path, f"{args.model}_autoaug-{preset}_{args.epoch}ep.pt")
            if os.path.exists(pt_path):
                os.remove(pt_path)

    rows_sorted = sorted(rows, key=lambda r: r['macro_f1'], reverse=True)
    with open(args.csv_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['aug', 'value', 'macro_f1', 'top1', 'top5', 'delta_f1'])
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(f"\n{'='*60}")
    print(f"[ablation] Resultado salvo em: {args.csv_out}")
    print(f"{'='*60}")
    print(f"{'augmentation':<15} {'macro_f1':>10} {'delta':>10}")
    for r in rows_sorted:
        print(f"{r['aug']:<15} {r['macro_f1']:>10.5f} {r['delta_f1']:>+10.5f}")


if __name__ == '__main__':
    main()