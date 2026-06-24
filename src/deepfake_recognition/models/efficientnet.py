"""EfficientNet-B3 fine-tuned for deepfake detection."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class DeepfakeEfficientNet(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.35,
                 freeze_backbone: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features   # 1536 for B3
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )
        if freeze_backbone:
            self._freeze()

    def _freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model(cfg: dict) -> DeepfakeEfficientNet:
    return DeepfakeEfficientNet(
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.35),
        freeze_backbone=cfg.get("freeze_backbone_epochs", 0) > 0,
    )
