"""
- CLASS-WEIGHTED LOSS: pondera o termo de cross-entropy de cada amostra
  pelo peso inverso da frequencia da SUA classe (classes raras pesam mais
  no gradiente). Nao muda quais imagens sao vistas nem com que frequencia, 
  so o quanto cada erro "custa" durante o backward.

- FOCAL LOSS: multiplica o termo de cross-entropy por
  (1 - p_t)**gamma, reduzindo a contribuicao de exemplos ja bem
  classificados (tipicamente a classe majoritaria / casos faceis) e
  concentrando o gradiente nos exemplos dificeis ou mal classificados
  (tipicamente as classes minoritarias e casos ambiguos

Os dois sao combinaveis (class-weighted focal loss)

Uso ja integrado em model/train.py via --gamma / --class-weighted)

"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.utils import LOGGER


class ClassBalancedFocalLoss:

    def __init__(self, class_weights: torch.Tensor | None = None, gamma: float = 0.0):
        self.gamma = gamma
        self.class_weights = class_weights

    def __call__(self, preds, batch: dict[str, torch.Tensor]):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        targets = batch["cls"]

        weight = self.class_weights.to(preds.device) if self.class_weights is not None else None
        ce = F.cross_entropy(preds, targets, weight=weight, reduction="none")

        if self.gamma > 0:
            with torch.no_grad():
                log_pt = -F.cross_entropy(preds, targets, weight=None, reduction="none")
                pt = log_pt.exp()
            focal_term = (1.0 - pt).clamp(min=0.0) ** self.gamma
            ce = focal_term * ce

        loss = ce.mean()
        return loss, {"loss": loss.detach()}


def _compute_class_weights(dataset, nc: int) -> torch.Tensor:
    counts = torch.zeros(nc)
    for sample in dataset.samples:
        cls_idx = sample[1]
        if cls_idx < nc:
            counts[cls_idx] += 1
    counts = counts.clamp(min=1)  #
    weights = 1.0 / counts
    weights = weights * (nc / weights.sum())
    return weights


def make_focal_loss_trainer(gamma: float = 2.0, class_weighted: bool = True) -> type[ClassificationTrainer]:

    class _FocalLossTrainer(ClassificationTrainer):
        def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)

            if mode == "train":
                nc = self.data.get("nc", 0)
                class_weights = _compute_class_weights(loader.dataset, nc) if class_weighted else None

                self.model.criterion = ClassBalancedFocalLoss(class_weights=class_weights, gamma=gamma)

                if class_weighted:
                    LOGGER.info(
                        f"[focal_loss] pesos por classe ativos "
                        f"(min={class_weights.min():.3f}, max={class_weights.max():.3f})."
                    )
                LOGGER.info(f"[focal_loss] gamma={gamma} (0 = sem termo focal, so cross-entropy).")

            return loader

    return _FocalLossTrainer