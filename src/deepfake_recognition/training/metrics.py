"""Metric tracking across batches."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._logits: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._losses: list[float] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float):
        self._logits.append(logits.detach().cpu())
        self._labels.append(labels.detach().cpu())
        self._losses.append(loss)

    def compute(self) -> dict[str, float]:
        all_logits = torch.cat(self._logits)
        all_labels = torch.cat(self._labels).numpy()
        probs = torch.softmax(all_logits, dim=1)[:, 1].numpy()
        preds = (probs > 0.5).astype(int)

        acc = (preds == all_labels).mean()
        try:
            auc = roc_auc_score(all_labels, probs)
        except Exception:
            auc = 0.0
        f1 = f1_score(all_labels, preds, zero_division=0)
        cm = confusion_matrix(all_labels, preds).tolist()
        avg_loss = np.mean(self._losses)

        return {
            "loss": float(avg_loss),
            "accuracy": float(acc),
            "auc": float(auc),
            "f1": float(f1),
            "confusion_matrix": cm,
        }
