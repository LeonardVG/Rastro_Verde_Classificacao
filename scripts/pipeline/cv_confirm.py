"""
Uso:
    python3 ./scripts/pipeline/cv_confirm.py \
        --model yolo26m-cls.pt \
        --epoch 20 \
        --k 5 \
        --train-path ./cls/dataset/train \
        --val-path ./cls/dataset/val \
        --models-path ./Models/Base \
        --save-models-path ./Models/CVConfirm \
        --result-path ./results/cv_confirm \
        --fold-output ./cls/dataset_kfold \
        --csv-out ./results/cv_confirm/summary.csv \
        --gamma 0     \
        --class-weighted   \
        --imgsz 512        \
        --aug-hsv_h 0.734136     \
        --aug-hsv_s 0.629738     \
        --aug-hsv_v 0.610808     \
        --aug-scale 0.610808     \
        --aug-fliplr 0.734136     \
        --aug-flipud 0.629738     \
        --aug-auto_augment 0.269499     \
        --aug-erasing 0.269499     \
        --lr0 0.027531          \
        --weight-decay 0.000723     \
        --dropout 0.236927 
"""
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset.kfold import gen as gen_kfold
from model.train import train
from model.bench import bench


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confirma a config vencedora do Optuna com k-fold")
    p.add_argument('--model', type=str, default='yolov8n-cls.pt')
    p.add_argument('--epoch', type=int, default=25)
    p.add_argument('--save-period', type=int, default=-1)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--imgsz', type=int, default=224)
    p.add_argument('--batch', type=int, default=16)

    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--class-weighted', action='store_true')

    p.add_argument('--train-path', type=str, required=True, help='train/ original (vai virar pool junto com val/)')
    p.add_argument('--val-path', type=str, required=True, help='val/ original (vai virar pool junto com train/)')
    p.add_argument('--fold-output', type=str, required=True, help='onde salvar os folds gerados')
    p.add_argument('--models-path', type=str, required=True)
    p.add_argument('--save-models-path', type=str, required=True)
    p.add_argument('--result-path', type=str, required=True)
    p.add_argument('--csv-out', type=str, required=True)

    p.add_argument('--hsv_h', type=float, default=0)
    p.add_argument('--hsv_s', type=float, default=0)
    p.add_argument('--hsv_v', type=float, default=0)
    p.add_argument('--scale', type=float, default=0)
    p.add_argument('--flipud', type=float, default=0.0)
    p.add_argument('--fliplr', type=float, default=0.0)
    p.add_argument('--auto_augment', type=str, default=None,
                        choices=['randaugment', 'autoaugment', 'augmix'])
    p.add_argument('--erasing', type=float, default=0)

    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--lr0', type=float, default=0.01)
    p.add_argument('--optimizer', type=str, default='AdamW',
                   help="mesmo padrão do core.py. Com 'auto' o Ultralytics ignora o --lr0")
    

    p.add_argument('--skip-fold-gen', action='store_true',
                    help='pula a geração dos folds (usa os que já existem em --fold-output)')
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    if not args.skip_fold_gen:
        gen_kfold(args.train_path, args.val_path, args.fold_output, args.k, args.seed)

    kwargs = {
        'hsv_h': args.hsv_h, 
        'hsv_s': args.hsv_s, 
        'hsv_v': args.hsv_v,
        'imgsz': args.imgsz,
        'auto_augment': None if args.auto_augment == 'none' else args.auto_augment,
        'fliplr': args.fliplr,
        'flipud': args.flipud,
        'scale': args.scale,
        'erasing': args.erasing,
        'lr0': args.lr0,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'gamma': args.gamma,
        'class_weighted': args.class_weighted,
        'batch': args.batch,
        'optimizer': args.optimizer,
    }

    # train() retoma de <result-path>/<nome>/weights/last.pt se ele existir.
    # Reusar o mesmo --result-path em outra config faria o fold partir do
    # modelo da config anterior. Aborta antes de treinar.
    for fold_idx in range(args.k):
        name = f"{args.model}_cv-fold{fold_idx}_{args.epoch}ep"
        ck = os.path.join(args.result_path, name, 'weights', 'last.pt')
        if os.path.exists(ck):
            raise SystemExit(f"[cv_confirm] ERRO: {ck} já existe. Use um --result-path novo para cada configuração.")

    rows = []
    for fold_idx in range(args.k):
        fold_root = os.path.join(args.fold_output, f"fold_{fold_idx}")
        fold_train = os.path.join(fold_root, "train")
        fold_val = os.path.join(fold_root, "val")

        print(f"\n{'='*60}\n[cv_confirm] fold {fold_idx}/{args.k - 1}\n{'='*60}")

        best_pt = train(args.model, args.epoch, args.save_period, args.models_path,
                         args.save_models_path, args.result_path, f"cv-fold{fold_idx}",
                         fold_train, fold_val, **kwargs)

        metrics = bench(best_pt, args.epoch, args.save_period, args.models_path,
                         args.save_models_path, args.result_path, fold_val, imgsz=args.imgsz)

        print(f"[cv_confirm] fold {fold_idx}: macro_f1={metrics['macro_f1']:.5f}")
        rows.append({'fold': fold_idx, 'macro_f1': metrics['macro_f1'],
                      'top1': metrics['top1'], 'top5': metrics['top5']})

    f1_values = [r['macro_f1'] for r in rows]
    mean_f1 = statistics.mean(f1_values)
    std_f1 = statistics.stdev(f1_values) if len(f1_values) > 1 else 0.0

    rows.append({'fold': 'MEAN', 'macro_f1': mean_f1, 'top1': '', 'top5': ''})
    rows.append({'fold': 'STD', 'macro_f1': std_f1, 'top1': '', 'top5': ''})

    with open(args.csv_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'macro_f1', 'top1', 'top5'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"[cv_confirm] macro-F1 médio: {mean_f1:.5f} ± {std_f1:.5f}")
    print(f"[cv_confirm] resultado salvo em: {args.csv_out}")
    print(f"{'='*60}")

    if std_f1 > 0.05:
        print("\n[cv_confirm] AVISO: desvio padrão alto entre folds (>0.05) ")


if __name__ == '__main__':
    main()