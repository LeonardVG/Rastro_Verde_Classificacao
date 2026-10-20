"""

Uso:
    python3 ./scripts/pipeline/optuna_search.py \
        --model yolo26m-cls.pt \
        --epoch 25 \
        --n-trials 50 \
        --train-path ./cls/dataset_Outros_clahe_1/train \
        --val-path ./cls/dataset_Outros_clahe_1/val \
        --models-path ./Models/Base \
        --save-models-path ./Models/apenas_outros/optuna \
        --result-path ./results/apenas_outros/optuna \
        --csv-out ./results/apenas_outros/optuna/trials.csv \
        --imgsz 512 \
        --gamma 0.0 \
        --class-weighted
        --batch 8 \
        --storage sqlite:///./results/apenas_outros/optuna/study.db

        
Teste rápido (1 trial só, poucas épocas, pra validar que está tudo funcionando):
    python3 ./scripts/pipeline/optuna_search.py \
        --model yolo26m-cls.pt \
        --train-path ./cls/dataset/train \
        --val-path ./cls/dataset/val \
        --models-path ./Models/Base \
        --save-models-path ./Models/Optuna_teste \
        --result-path ./results/optuna_teste \
        --csv-out ./results/optuna_teste/trials_smoke.csv \
        --smoke-test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.train import train
from model.bench import bench

import optuna

def build_kwargs(trial: optuna.Trial, gamma: float, class_weighted: bool) -> dict:
    kwargs = {'gamma': gamma, 'class_weighted': class_weighted}

    # Augmentation
    kwargs['hsv_h'] = trial.suggest_float('hsv_h', 0.0, 0.1)
    kwargs['hsv_s'] = trial.suggest_float('hsv_s', 0.0, 1.0)
    kwargs['hsv_v'] = trial.suggest_float('hsv_v', 0.0, 1.0)
    kwargs['scale'] = trial.suggest_float('scale', 0.0, 0.9)
    kwargs['fliplr'] = trial.suggest_float('fliplr', 0.0, 1.0)
    kwargs['flipud'] = trial.suggest_float('flipud', 0.0, 1.0)
    kwargs['auto_augment'] = trial.suggest_categorical('auto_augment', [None, 'randaugment', 'autoaugment', 'augmix'])
    kwargs['erasing'] = trial.suggest_float('erasing', 0.0, 1.0)

    # Hiperparâmetros
    kwargs['lr0'] = trial.suggest_float('lr0', 1e-4, 1e-1, log=True)
    kwargs['weight_decay'] = trial.suggest_float('weight_decay',  1e-6, 1e-3, log=True)
    kwargs['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)

    # busca local
    #kwargs['hsv_h'] = trial.suggest_float('hsv_h', 0.035, 0.075)
    #kwargs['hsv_s'] = trial.suggest_float('hsv_s', 0.25, 0.52)
    #kwargs['hsv_v'] = trial.suggest_float('hsv_v', 0.20, 0.42)
    #kwargs['scale'] = trial.suggest_float('scale', 0.32, 0.57)
    #kwargs['fliplr'] = trial.suggest_float('fliplr', 0.10, 0.42)
    #kwargs['flipud'] = trial.suggest_float('flipud', 0.62, 0.92)
    #kwargs['auto_augment'] = 'augmix'
    #kwargs['erasing'] = trial.suggest_float('erasing', 0.15, 0.38)


    #kwargs['lr0'] = trial.suggest_float('lr0', 1.5e-4, 7e-4, log=True)
    #kwargs['weight_decay'] = trial.suggest_float('weight_decay', 3e-6, 2.5e-5, log=True)
    #kwargs['dropout'] = trial.suggest_float('dropout', 0.22, 0.44)
    
    return kwargs


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Busca de hiperparâmetros com Optuna")
    p.add_argument('--model', type=str, default='yolov8n-cls.pt')
    p.add_argument('--epoch', type=int, default=15, help='épocas por trial (mantenha baixo, é só pra ranquear)')
    p.add_argument('--imgsz', type=int, default=224,
                   help='resolução das imagens usada no treino e validação')
    p.add_argument('--batch', type=int, default=16,
                   help='batch size usado no treino')
    p.add_argument('--save-period', type=int, default=-1)
    p.add_argument('--n-trials', type=int, default=30)
    p.add_argument('--train-path', type=str, required=True)
    p.add_argument('--val-path', type=str, required=True)
    p.add_argument('--models-path', type=str, required=True)
    p.add_argument('--save-models-path', type=str, required=True)
    p.add_argument('--result-path', type=str, required=True)
    p.add_argument('--csv-out', type=str, required=True)
    p.add_argument('--keep-checkpoints', action='store_true',
                    help='se não passar, mantém só o checkpoint do melhor trial até agora')
    p.add_argument('--study-name', type=str, default='plantas_augmentation_search')
    p.add_argument('--storage', type=str, default=None,
                    help='ex: sqlite:///../Results/optuna/study.db — se passado, salva progresso no disco '
                         'e permite retomar rodando o mesmo comando de novo depois de uma interrupção')
    p.add_argument('--smoke-test', action='store_true',
                    help='roda 1 trial só, com poucas épocas, pra validar o pipeline')
    p.add_argument('--gamma', type=float, default=0.0,
                    help='gamma do focal loss, fixo durante essa busca (escolhido no grid search prévio)')
    p.add_argument('--class-weighted', action='store_true',
                    help='usa loss ponderada por classe, fixo durante essa busca (escolhido no grid search prévio)')
    p.add_argument('--optimizer', type=str, default='AdamW',
                    help='otimizador fixo para permitir que o Optuna ajuste o lr0')
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    if args.smoke_test:
        args.n_trials = 1
        args.epoch = min(args.epoch, 2)
        print(f"[optuna] SMOKE TEST: n_trials=1, epoch={args.epoch}\n")

    best_f1_so_far = {'value': -1.0, 'pt_path': None}

    def objective(trial: optuna.Trial) -> float:
        kwargs = build_kwargs(trial, gamma=args.gamma, class_weighted=args.class_weighted)
        kwargs['imgsz'] = args.imgsz
        kwargs['batch'] = args.batch
        kwargs['optimizer'] = args.optimizer
        dataset_name = f"optuna-trial{trial.number}"

        print(f"\n{'='*60}\n[optuna] trial {trial.number}: {trial.params}\n{'='*60}")

        def on_epoch_end(step: int, metrics_epoch: dict):
            macro_f1 = metrics_epoch['macro_f1']
            accuracy = metrics_epoch.get('top1')
            acc_str = f", val_accuracy={float(accuracy):.5f}" if accuracy is not None else ""

            trial.report(macro_f1, step=step)
            print(f"[optuna] época {step + 1}: val_macro_f1={macro_f1:.5f}{acc_str}")

            if trial.should_prune():
                print(f"[optuna] trial {trial.number} podada na época {step + 1} "
                      f"(val_macro_f1={macro_f1:.5f})")
                raise optuna.TrialPruned()

        best_pt = train(
            args.model,
            args.epoch,
            args.save_period,
            args.models_path,
            args.save_models_path,
            args.result_path,
            dataset_name,
            args.train_path,
            args.val_path,
            on_epoch_end=on_epoch_end,
            **kwargs,
        )

        metrics = bench(best_pt, args.epoch, args.save_period, args.models_path,
                         args.save_models_path, args.result_path, args.val_path, imgsz=args.imgsz)

        macro_f1 = metrics['macro_f1']
        top1 = metrics['top1']
        trial.set_user_attr('top1', top1)

        print(f"[optuna] trial {trial.number}: macro_f1={macro_f1:.5f}, top1={top1:.5f}")

        if not args.keep_checkpoints:
            if macro_f1 > best_f1_so_far['value']:
                if best_f1_so_far['pt_path'] and os.path.exists(best_f1_so_far['pt_path']):
                    os.remove(best_f1_so_far['pt_path'])
                best_f1_so_far['value'] = macro_f1
                best_f1_so_far['pt_path'] = os.path.join(
                    args.save_models_path, f"{args.model}_{dataset_name}_{args.epoch}ep.pt"
                )
            else:
                pt_path = os.path.join(args.save_models_path, f"{args.model}_{dataset_name}_{args.epoch}ep.pt")
                if os.path.exists(pt_path):
                    os.remove(pt_path)

        return macro_f1

    if args.storage and args.storage.startswith('sqlite:///'):
        db_path = args.storage.replace('sqlite:///', '', 1)
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=6,
        interval_steps=1,
    )

    study = optuna.create_study(direction='maximize', study_name=args.study_name,
                                 storage=args.storage, load_if_exists=True,
                                 pruner=pruner)
    n_done = len(study.trials)
    if n_done > 0:
        print(f"[optuna] retomando study existente: {n_done} trials já rodados anteriormente")
    else:
        study.enqueue_trial({
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'scale': 0.5,
            'fliplr': 0.5,
            'flipud': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'dropout': 0.0,
        })
        print("[optuna] baseline (valores default) enfileirado como primeira trial")

    study.optimize(objective, n_trials=args.n_trials)

    study.trials_dataframe().to_csv(args.csv_out, index=False)

    print(f"\n{'='*60}")
    print(f"[optuna] Resultado salvo em: {args.csv_out}")
    print(f"[optuna] Melhor macro-F1 (sem filtro de acurácia): {study.best_value:.5f}")
    print(f"[optuna] Melhores parâmetros (por macro-F1):")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    ACC_THRESHOLD = 0.95
    completed = [t for t in study.trials if t.value is not None]
    passing = [t for t in completed if t.user_attrs.get('top1', 0.0) >= ACC_THRESHOLD]

    print(f"\n[optuna] Trials com acurácia >= {ACC_THRESHOLD:.0%}: {len(passing)}/{len(completed)}")
    if passing:
        best_passing = max(passing, key=lambda t: t.value)
        print(f"[optuna] Melhor macro-F1 entre os que batem {ACC_THRESHOLD:.0%} de acurácia: "
              f"trial {best_passing.number} (macro_f1={best_passing.value:.5f}, "
              f"top1={best_passing.user_attrs.get('top1'):.5f})")
        print(f"[optuna] Parâmetros desse trial:")
        for k, v in best_passing.params.items():
            print(f"    {k}: {v}")
    else:
        print(f"[optuna] Nenhum trial bateu {ACC_THRESHOLD:.0%} de acurácia — "
              f"revise o CSV completo em {args.csv_out}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()