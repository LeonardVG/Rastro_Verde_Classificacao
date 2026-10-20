from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO
import numpy as np


def bench(model: str, epoch: int, save_period: int, models_path: str, msave_path: str, result_path: str,
          test_path: str, **kwargs) -> dict:
    imgsz = kwargs.get('imgsz', 224)

    if os.path.exists(model):
        p = Path(model)
        name = p.parent.parent.parent.name if p.parent.name == 'weights' else p.stem
        model_path = model
    else:
        name = model.replace('.pt', '')
        model_path = os.path.join(
            models_path,
            model if model.endswith('.pt') else f"{model}.pt"
        )

    yolo = YOLO(model_path, task='classify')

    test_path_clean = test_path.rstrip('/\\')
    data = os.path.dirname(test_path_clean)
    split = os.path.basename(test_path_clean)

    save = os.path.join(result_path, name)
    os.makedirs(save, exist_ok=True)

    metrics = yolo.val(
        data=data,
        split=split,
        project=result_path,
        name=name,
        exist_ok=True,
        plots=True,
        imgsz=imgsz
    )

    macro_f1 = 0.0

    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix.matrix.any():
        cm_matrix = metrics.confusion_matrix.matrix

        tp = np.diag(cm_matrix)
        fp = cm_matrix.sum(axis=0) - tp
        fn = cm_matrix.sum(axis=1) - tp

        precision = np.divide(
            tp,
            tp + fp,
            out=np.zeros_like(tp, dtype=float),
            where=(tp + fp) != 0
        )

        recall = np.divide(
            tp,
            tp + fn,
            out=np.zeros_like(tp, dtype=float),
            where=(tp + fn) != 0
        )

        f1_per_class = 2 * (precision * recall) / (precision + recall)
        macro_f1 = np.mean(np.nan_to_num(f1_per_class))

    print(f"'Macro F1-Score': {macro_f1:.5f}")
    print(f"'Acc (T1)': {metrics.top1:.5f}")
    print(f"'Acc (T5)': {metrics.top5:.5f}")

    for key, value in metrics.speed.items():
        print(f"'{key}': {value}")

    print(
        f"'fps': "
        f"{1000 / metrics.speed['inference'] if metrics.speed['inference'] > 0 else 0:.5f}"
    )

    os.makedirs(msave_path, exist_ok=True)
    yolo.save(os.path.join(msave_path, f"{name}.pt"))

    with open(os.path.join(save, "results.txt"), 'w') as res:
        res.write(f"'Macro F1-Score': {macro_f1:.3f}\n")
        res.write(f"{metrics.speed}\n")
        res.write(f"'Acc (T1)': {metrics.top1:.5f}\n")
        res.write(f"'Acc (T5)': {metrics.top5:.5f}\n")
        res.write(
            f"'fps': "
            f"{1000 / metrics.speed['inference'] if metrics.speed['inference'] > 0 else 0}\n"
        )

    return {
        'macro_f1': macro_f1,
        'top1': metrics.top1,
        'top5': metrics.top5
    }