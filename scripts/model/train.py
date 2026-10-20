import os
from typing import Callable, Optional
from ultralytics import YOLO
import numpy as np
import csv
import tempfile

from dataset.loss import make_focal_loss_trainer


def macro_f1_from_confusion(cm_matrix: np.ndarray) -> float:
    tp = np.diag(cm_matrix)
    fp = cm_matrix.sum(axis=0) - tp
    fn = cm_matrix.sum(axis=1) - tp

    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0
    )

    f1_per_class = 2 * (precision * recall) / (precision + recall)
    return float(np.mean(np.nan_to_num(f1_per_class)))


def registrar_macro_f1_csv(caminho_csv: str, epoca: int, macro_f1: float) -> None:
    """
    Adiciona ou atualiza o Macro F1 da época no results.csv
    gerado pelo Ultralytics.
    """
    if not os.path.isfile(caminho_csv):
        return

    with open(caminho_csv, "r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        campos = list(leitor.fieldnames or [])
        linhas = list(leitor)

    if not linhas or "epoch" not in campos:
        return

    coluna = "metrics/macro_f1"

    if coluna not in campos:
        campos.append(coluna)

    for linha in linhas:
        try:
            numero_epoca = int(float(linha["epoch"]))
        except (ValueError, TypeError):
            continue

        if numero_epoca == epoca + 1:
            linha[coluna] = f"{macro_f1:.6f}"

    diretorio = os.path.dirname(caminho_csv) or "."
    arquivo_temporario = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=diretorio,
            delete=False,
            suffix=".tmp",
        ) as arquivo:
            arquivo_temporario = arquivo.name
            escritor = csv.DictWriter(arquivo, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(linhas)

        os.replace(arquivo_temporario, caminho_csv)

    finally:
        if arquivo_temporario and os.path.exists(arquivo_temporario):
            os.remove(arquivo_temporario)


def adicionar_macro_f1_csv(caminho_csv: str, f1_por_epoca: dict) -> None:
    import pandas as pd

    if not os.path.isfile(caminho_csv):
        raise FileNotFoundError(f"CSV não encontrado: {caminho_csv}")

    if not f1_por_epoca:
        raise ValueError("Nenhum Macro F1 foi armazenado.")

    df = pd.read_csv(caminho_csv)
    df["epoch"] = df["epoch"].astype(int)

    # Mantém a correspondência com trainer.epoch, que começa em zero.
    df["metrics/macro_f1"] = df["epoch"].map(f1_por_epoca)

    df.to_csv(caminho_csv, index=False)

    df_verificacao = pd.read_csv(caminho_csv)

    if "metrics/macro_f1" not in df_verificacao.columns:
        raise RuntimeError("A coluna Macro F1 não foi gravada.")

    print(f"[F1] CSV atualizado: {caminho_csv}")


def gerar_graficos_personalizados(caminho_csv: str) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    if not os.path.isfile(caminho_csv):
        print(f"[Gráficos] CSV não encontrado: {caminho_csv}")
        return

    df = pd.read_csv(caminho_csv)
    df.columns = df.columns.str.strip()

    colunas_necessarias = {"epoch", "metrics/macro_f1"}
    if not colunas_necessarias.issubset(df.columns):
        print("[Gráficos] Faltam colunas necessárias no CSV.")
        return

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["metrics/macro_f1"] = pd.to_numeric(
        df["metrics/macro_f1"], errors="coerce"
    )

    pasta = os.path.dirname(caminho_csv)

    # Macro F1-score
    dados_f1 = df.dropna(subset=["epoch", "metrics/macro_f1"])

    if not dados_f1.empty:
        plt.figure(figsize=(9, 5))
        plt.plot(
            dados_f1["epoch"],
            dados_f1["metrics/macro_f1"],
            marker="o",
            label="Macro F1-score",
        )
        plt.xlabel("Época")
        plt.ylabel("Macro F1-score")
        plt.title("Macro F1-score por época")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        caminho_f1 = os.path.join(pasta, "macro_f1.png")
        plt.savefig(caminho_f1, dpi=150)
        plt.close()
        print(f"[Gráficos] Salvo: {caminho_f1}")

    # Loss de treino e validação
    colunas_loss = ["train/loss", "val/loss"]

    if all(coluna in df.columns for coluna in colunas_loss):
        dados_loss = df.dropna(subset=["epoch"] + colunas_loss)

        if not dados_loss.empty:
            plt.figure(figsize=(9, 5))
            plt.plot(
                dados_loss["epoch"],
                dados_loss["train/loss"],
                marker="o",
                label="Loss de treino",
            )
            plt.plot(
                dados_loss["epoch"],
                dados_loss["val/loss"],
                marker="s",
                label="Loss de validação",
            )
            plt.xlabel("Época")
            plt.ylabel("Loss")
            plt.title("Loss de treino e validação")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            caminho_loss = os.path.join(
                pasta, "loss_treino_validacao.png"
            )
            plt.savefig(caminho_loss, dpi=150)
            plt.close()
            print(f"[Gráficos] Salvo: {caminho_loss}")
    else:
        print(
            "[Gráficos] Colunas de loss não encontradas. "
            f"Colunas disponíveis: {list(df.columns)}"
        )


def train(
    model: str,
    epoch: int,
    save_period: int,
    models_path: str,
    msave_path: str,
    result_path: str,
    dataset_name: str,
    train_path: str,
    val_path: str,
    gamma: float = 0.0,
    class_weighted: bool = False,
    on_epoch_end: Optional[Callable[[int, dict], None]] = None,
    **kwargs,
) -> None:

    data = val_path.rsplit(os.sep, maxsplit=1)[0]
    ds_name = dataset_name
    name = f"{model}_{ds_name.lower()}_{epoch}ep"

    if not model.endswith(".pt"):
        model = f"{model}.pt"

    check_point = os.path.join(
        result_path, f"{name}", "weights", "last.pt"
    )

    if os.path.exists(check_point):
        yolo = YOLO(check_point)
    else:
        yolo = YOLO(os.path.join(models_path, model))

    trainer_cls = None

    if gamma > 0 or class_weighted:
        trainer_cls = make_focal_loss_trainer(
            gamma=gamma,
            class_weighted=class_weighted,
        )

    f1_por_epoca = {}

    def _on_fit_epoch_end(trainer):
        validator = getattr(trainer, "validator", None)
        val_metrics = getattr(validator, "metrics", None)

        cm = (
            getattr(val_metrics, "confusion_matrix", None)
            if val_metrics is not None
            else None
        )
        cm_matrix = getattr(cm, "matrix", None) if cm is not None else None

        if cm_matrix is None or not cm_matrix.any():
            print("[F1] Matriz de confusão indisponível nesta época.")
            return

        step = int(getattr(trainer, "epoch", 0))
        macro_f1 = macro_f1_from_confusion(cm_matrix)
        accuracy = getattr(val_metrics, "top1", None)

        f1_por_epoca[step] = macro_f1

        print(f"[F1] Época {step + 1}: Macro F1 = {macro_f1:.5f}")

        if on_epoch_end is not None:
            on_epoch_end(
                step,
                {
                    "macro_f1": macro_f1,
                    "top1": accuracy,
                },
            )

    yolo.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    # Garante o uso dos parâmetros recebidos, inclusive lr0 e optimizer.
    # Os demais argumentos continuam sendo repassados normalmente.
    kwargs_treino = {
        **kwargs,
        "data": data,
        "epochs": epoch,
        "project": os.path.join(result_path, name),
        "save_period": save_period,
        "workers": 2,
    }

    # Para treinamento sem aumento de dados, preserva as configurações
    # originais e mantém os parâmetros de otimização recebidos.
    if ds_name.lower() == "noaug":
        kwargs_treino.update({
            "augment": False,
            "hsv_h": 0,
            "hsv_s": 0,
            "hsv_v": 0,
            "degrees": 0.0,
            "translate": 0,
            "scale": 0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0,
            "mosaic": 0,
            "mixup": 0.0,
            "auto_augment": None,
            "erasing": 0,
        })

    # trainer personalizado somente quando necessário para a loss.
    if trainer_cls is not None:
        kwargs_treino["trainer"] = trainer_cls
    # Com optimizer=auto o Ultralytics zera o warmup_bias_lr; com AdamW explícito
    # ele fica em 0.1 (pensado para SGD) e desestabiliza o início do treino.
    if str(kwargs_treino.get("optimizer", "auto")).lower() not in ("auto", "sgd"):
        kwargs_treino.setdefault("warmup_bias_lr", 0.0)

    print(
        f"[Treino] optimizer={kwargs_treino.get('optimizer', 'auto')}, "
        f"lr0={kwargs_treino.get('lr0', 'padrão')}"
    )

    yolo.train(**kwargs_treino)

    caminho_csv = os.path.join(
        str(yolo.trainer.save_dir),
        "results.csv",
    )

    adicionar_macro_f1_csv(caminho_csv, f1_por_epoca)
    gerar_graficos_personalizados(caminho_csv)

    exported_path = yolo.export()
    print(f"Modelo exportado para: {exported_path}")

    best_pt = str(yolo.trainer.save_dir / "weights" / "best.pt")
    print(f"Melhor checkpoint: {best_pt}")

    return best_pt
