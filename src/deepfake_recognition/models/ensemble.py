"""Weighted soft-voting ensemble of deepfake detection models."""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F


class EnsemblePredictor:
    """
    Loads multiple checkpoints and combines their softmax outputs
    via weighted averaging.

    Config example (from training/configs/efficientnet_b3.yaml):
        ensemble:
          models:
            - checkpoint: checkpoints/resnet18/best.pth
              weight: 0.40
            - checkpoint: checkpoints/efficientnet_b3/best.pth
              weight: 0.60
          tta_enabled: true
    """

    def __init__(self, members: list[dict], device: str = "auto"):
        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.predictors = []
        self.weights = []

        from deepfake_recognition.inference.predictor import Predictor
        for m in members:
            ckpt = Path(m["checkpoint"])
            if not ckpt.exists():
                raise FileNotFoundError(f"Ensemble checkpoint not found: {ckpt}")
            self.predictors.append(Predictor.from_checkpoint(ckpt, device=device))
            self.weights.append(m.get("weight", 1.0))

        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def predict_pil(self, img, use_tta: bool = False) -> dict:
        from PIL import Image
        all_probs = []
        for predictor, weight in zip(self.predictors, self.weights):
            r = predictor.predict_pil(img, use_tta=use_tta)
            all_probs.append(
                torch.tensor([r["prob_real"], r["prob_fake"]]) * weight
            )
        avg = torch.stack(all_probs).sum(0)
        prob_real, prob_fake = avg[0].item(), avg[1].item()
        return {
            "label": "fake" if prob_fake > 0.5 else "real",
            "confidence": round(max(prob_real, prob_fake), 4),
            "prob_real": round(prob_real, 4),
            "prob_fake": round(prob_fake, 4),
            "gradcam_image": None,  # ensemble GradCAM not supported
        }

    @classmethod
    def from_config(cls, cfg: dict, device: str = "auto") -> "EnsemblePredictor":
        return cls(members=cfg["ensemble"]["models"], device=device)
