"""
Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) como
pré-processamento FIXO no dataset

roda no canal de luminância (LAB), preservando cor

Uso:
    python clahe.py --dataset-root ./dataset_original \
        --preview --preview-n 5

    python ./scripts/dataset/clahe.py --dataset-root ./cls/dataset_inteiro/val \
        --output-root ./cls/dataset_inteiro_clahe/val --clip-limit 1.0

"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLITS = ("train", "val", "test")


def apply_clahe(img_bgr: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def find_splits(dataset_root: Path) -> list[Path]:
    found = [dataset_root / s for s in DEFAULT_SPLITS if (dataset_root / s).is_dir()]
    return found if found else [dataset_root]


def list_class_dirs(split_dir: Path) -> list[Path]:
    return sorted(p for p in split_dir.iterdir() if p.is_dir())


def list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def run_preview(dataset_root: Path, out_dir: Path, clip_limit: float, tile_grid_size: tuple[int, int], n_per_class: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = find_splits(dataset_root)
    split_dir = splits[0]

    print(f"Gerando preview (original | CLAHE) em: {out_dir}\n")
    for class_dir in list_class_dirs(split_dir):
        images = list_images(class_dir)[:n_per_class]
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            eq = apply_clahe(img, clip_limit, tile_grid_size)

            side_by_side = np.hstack([img, eq])
            out_name = f"{class_dir.name}__{img_path.stem}_compare.jpg"
            cv2.imwrite(str(out_dir / out_name), side_by_side)

        if images:
            print(f"  {class_dir.name}: {len(images)} preview(s) gerado(s)")

    print(f"\nImagens em {out_dir} (metade esquerda = original, direita = com CLAHE).")


def run_full(dataset_root: Path, output_root: Path, clip_limit: float, tile_grid_size: tuple[int, int]):
    splits = find_splits(dataset_root)

    for split_dir in splits:
        rel_split = split_dir.relative_to(dataset_root) if split_dir != dataset_root else Path(".")
        for class_dir in list_class_dirs(split_dir):
            out_class_dir = output_root / rel_split / class_dir.name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            images = list_images(class_dir)
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  [aviso] não consegui ler, pulando: {img_path}")
                    continue
                eq = apply_clahe(img, clip_limit, tile_grid_size)
                cv2.imwrite(str(out_class_dir / img_path.name), eq)

            print(f"{str(rel_split / class_dir.name):45s} {len(images):4d} imagens processadas")

    print(f"\nPronto. Dataset com CLAHE em: {output_root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True,
                     help="pasta com train/val/test (ou pasta de classes direto, se não houver splits)")
    ap.add_argument("--output-root", type=Path,
                     help="onde escrever o dataset processado (obrigatório fora do modo --preview)")
    ap.add_argument("--clip-limit", type=float, default=3.0, help="limite de contraste do CLAHE")
    ap.add_argument("--tile-size", type=int, default=8, help="tamanho do grid de tiles, NxN ")
    ap.add_argument("--preview", action="store_true", help="gera comparações lado a lado numa amostra, não processa tudo")
    ap.add_argument("--preview-n", type=int, default=5, help="quantas imagens por classe no preview")
    ap.add_argument("--preview-dir", type=Path, default=Path("./clahe_preview"))
    args = ap.parse_args()

    tile_grid_size = (args.tile_size, args.tile_size)

    if args.preview:
        run_preview(args.dataset_root, args.preview_dir, args.clip_limit, tile_grid_size, args.preview_n)
        return

    if not args.output_root:
        ap.error("--output-root é obrigatório fora do modo --preview (nunca sobrescrevemos os originais por padrão).")

    run_full(args.dataset_root, args.output_root, args.clip_limit, tile_grid_size)


if __name__ == "__main__":
    main()