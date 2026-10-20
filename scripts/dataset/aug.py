"""
Augmentation OFFLINE aplicada só nas classes minoritárias dosplit de TREINO.

As transformações foram escolhidas pra imitar variação real de campo
(plantas inteiras, ambiente natural), não augmentation agressivo de cor/
geometria que descaracteriza a planta:

- Iluminação/sombra: brightness-contrast, gamma, sombra simulada
- Ângulo/distância de câmera: rotação leve, perspectiva leve, escala
- Ruído de sensor em luz variável: leve gauss/ISO noise
- Desfoque de movimento/distância: motion blur leve
- Oclusão parcial (galhos, folhas na frente): coarse dropout leve

Uso:
    pip install albumentations opencv-python-headless --break-system-packages

    python ./scripts/dataset/aug.py \
            --data-dir ./cls/dataset/train \
            --classes Campomanesia_phaea  \
            --target-count 160
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

AUG_SUFFIX = "_aug" 


def build_pipeline(strength: str = "moderate") -> A.Compose:
    ranges = {
        "light":    dict(bc=0.15, hue=8,  sat=15, rot=8,  scale=0.08, blur=3, drop=0.03),
        "moderate": dict(bc=0.25, hue=12, sat=25, rot=15, scale=0.15, blur=5, drop=0.06),
        "strong":   dict(bc=0.35, hue=18, sat=35, rot=25, scale=0.25, blur=7, drop=0.10),
    }[strength]

    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=ranges["bc"], contrast_limit=ranges["bc"], p=0.8
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.HueSaturationValue(
            hue_shift_limit=ranges["hue"], sat_shift_limit=ranges["sat"],
            val_shift_limit=10, p=0.5,
        ),
        A.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_limit=(1, 2), p=0.25),

        A.Affine(
            rotate=(-ranges["rot"], ranges["rot"]),
            scale=(1 - ranges["scale"], 1 + ranges["scale"]),
            translate_percent=(0.0, 0.05),
            p=0.8,
        ),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.HorizontalFlip(p=0.5),

        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
        ], p=0.3),
        A.MotionBlur(blur_limit=(3, ranges["blur"]), p=0.2),

        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.05, ranges["drop"]),
            hole_width_range=(0.05, ranges["drop"]),
            fill=0,
            p=0.3,
        ),
    ])


def list_source_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS and AUG_SUFFIX not in p.stem
    )


def augment_class(
    class_dir: Path, target_count: int, pipeline: A.Compose,
    seed: int, dry_run: bool,
) -> int:
    sources = list_source_images(class_dir)
    current_total = len(list(class_dir.iterdir()))  
    n_needed = target_count - current_total

    if n_needed <= 0 or not sources:
        return 0

    rng = random.Random(seed)
    for i in range(n_needed):
        src = sources[i % len(sources)]  
        img = cv2.imread(str(src))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = pipeline(image=img)["image"]

        out_name = f"{src.stem}{AUG_SUFFIX}{i:03d}{src.suffix}"
        out_path = class_dir / out_name

        if not dry_run:
            augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), augmented)

    return n_needed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="pasta train/ (com uma subpasta por classe)")
    ap.add_argument("--classes", nargs="+", required=True, help="nomes das classes minoritárias a aumentar")
    ap.add_argument("--target-count", type=int, required=True, help="total de imagens desejado por classe (originais + geradas)")
    ap.add_argument("--strength", choices=["light", "moderate", "strong"], default="moderate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="só mostra quantas imagens seriam geradas, não grava nada")
    args = ap.parse_args()

    pipeline = build_pipeline(args.strength)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Alvo: {args.target_count} imagens/classe | força: {args.strength}\n")

    for cls_name in args.classes:
        class_dir = args.data_dir / cls_name
        if not class_dir.is_dir():
            print(f"  [aviso] pasta não encontrada, pulando: {class_dir}")
            continue

        before = len(list(class_dir.iterdir()))
        generated = augment_class(class_dir, args.target_count, pipeline, args.seed, args.dry_run)
        after = before + generated

        print(f"{cls_name:30s} {before:4d} -> {after:4d}  (+{generated})")

    print("\nPronto." if not args.dry_run else "\nDry run concluído -- nada foi gravado em disco.")


if __name__ == "__main__":
    main()