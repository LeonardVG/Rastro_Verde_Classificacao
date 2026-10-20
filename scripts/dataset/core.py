import argparse
import logging

from .cls import CLS
from os import path, getcwd


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset generator (classification)")

    parser.add_argument('--split', type=float, nargs=2, metavar=("val", "test"), required=True,
                        help='percentage of the dataset for validation and test set, the remaining is on training set')

    parser.add_argument('--split-out', type=str, default=path.join(getcwd(), "..", "cls", "dataset"),
                        help='Path to output the dataset (train/val/test folders, one subfolder per class each)')

    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path for the original dataset folder, organized as one subfolder per class (e.g., ./dataset_plantas_raw/)')

    return parser.parse_args()


def run() -> None:
    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(threadName)s] %(levelname)s: %(message)s',
    )

    logging.getLogger().setLevel(logging.INFO)

    CLS.gen(args.dataset_path, args.split_out, args.split[0], args.split[1])
