import argparse
import logging
from typing import *

from .bench import bench
from .train import train
from os import path, getcwd


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classification model trainer/benchmark")

    parser.add_argument("model", type=str,
                        help='model to use, must be available in --models-path (e.g. yolo11n-cls)')

    parser.add_argument("--bench", action='store_true',
                        help='benchmarks the model')

    parser.add_argument("--train", action='store_true',
                        help="train the model")

    parser.add_argument("--epoch", type=int, default=100,
                        help='epochs')

    parser.add_argument("--save-period", type=int, default=5,
                        help='save period')

    parser.add_argument('--models-path', type=str, default=path.join(getcwd(), "..", "Models", "Base"),
                        help='path for base models')

    parser.add_argument('--save-models-path', type=str, default=path.join(getcwd(), "..", "Models", "Trained"),
                        help='path to save trained model')

    parser.add_argument('--result-path', type=str, default=path.join(getcwd(), "..", "Results"),
                        help='path to output training/benchmark results')

    parser.add_argument('--dataset-name', type=str, default='Plantas',
                        help='dataset name, used to name the training run folder')

    parser.add_argument('--train-path', type=str, default=path.join(getcwd(), "..", "cls", "dataset", 'train'),
                        help='train dataset path (one subfolder per class)')

    parser.add_argument('--val-path', type=str, default=path.join(getcwd(), "..", "cls", "dataset", "val"),
                        help='val dataset path (one subfolder per class)')

    parser.add_argument('--test-path', type=str, default=path.join(getcwd(), "..", "cls", "dataset", "test"),
                        help='test dataset path (one subfolder per class), used for --bench')

    parser.add_argument('--gamma', type=float, default=0.0,
                        help='forca do termo focal na loss (0 = sem focal loss, so cross-entropy; '
                             'tipico 1.0-2.0, conforme Lin et al. 2017)')

    parser.add_argument('--class-weighted', action='store_true',
                        help='pondera a cross-entropy pelo inverso da frequencia de cada classe')

    parser.add_argument('--batch', type=int, default=16,
                        help='batch size (-1 para auto-batch baseado na memória livre da GPU)')

    parser.add_argument('--patience', type=int, default=100,
                        help='early stopping: épocas sem melhora antes de parar')

    # NOVO: define explicitamente o otimizador
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='otimizador utilizado no treinamento (ex: AdamW, SGD)')

    parser.add_argument('--aug-hsv_h', type=float, default=0)
    parser.add_argument('--aug-hsv_s', type=float, default=0)
    parser.add_argument('--aug-hsv_v', type=float, default=0)
    parser.add_argument('--aug-degrees', type=float, default=0.0)
    parser.add_argument('--aug-translate', type=float, default=0)
    parser.add_argument('--aug-scale', type=float, default=0)
    parser.add_argument('--aug-shear', type=float, default=0)
    parser.add_argument('--aug-perspective', type=float, default=0.0)
    parser.add_argument('--aug-flipud', type=float, default=0.0)
    parser.add_argument('--aug-fliplr', type=float, default=0.0)
    parser.add_argument('--aug-mosaic', type=float, default=0)
    parser.add_argument('--aug-mixup', type=float, default=0)
    parser.add_argument('--aug-cutmix', type=float, default=0)
    parser.add_argument('--aug-bgr', type=float, default=0)
    parser.add_argument('--aug-auto_augment', type=str, default=None,
                        choices=['randaugment', 'autoaugment', 'augmix'])
    parser.add_argument('--aug-erasing', type=float, default=0)

    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate')

    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='optimizer weight decay')

    parser.add_argument('--lr0', type=float, default=0.001,
                        help='initial learning rate')

    parser.add_argument('--imgsz', type=int, default=224,
                        help='resolução da imagem de entrada (ex: 224, 320, 384)')

    return parser.parse_args()


def run() -> None:
    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(threadName)s] %(levelname)s: %(message)s',
    )

    logging.getLogger().setLevel(logging.INFO)

    extra_args = {}

    for k, v in args._get_kwargs():
        if k.startswith('aug_'):
            extra_args[k.replace('aug_', '')] = v

    extra_args['dropout'] = args.dropout
    extra_args['weight_decay'] = args.weight_decay
    extra_args['lr0'] = args.lr0
    extra_args['optimizer'] = args.optimizer  # NOVO
    extra_args['imgsz'] = args.imgsz
    extra_args['batch'] = args.batch
    extra_args['patience'] = args.patience

    best_pt = None

    if args.train:
        best_pt = train(
            args.model,
            args.epoch,
            args.save_period,
            args.models_path,
            args.save_models_path,
            args.result_path,
            args.dataset_name,
            args.train_path,
            args.val_path,
            gamma=args.gamma,
            class_weighted=args.class_weighted,
            **extra_args
        )

    if args.bench:
        bench_model = best_pt if best_pt else args.model

        bench(
            bench_model,
            args.epoch,
            args.save_period,
            args.models_path,
            args.save_models_path,
            args.result_path,
            args.test_path,
            **extra_args
        )
