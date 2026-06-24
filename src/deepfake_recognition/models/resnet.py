"""ResNet-18 fine-tuned for deepfake detection."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class DeepfakeResNet18(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3,
                 freeze_backbone: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)
        # Remove avgpool and fc, keep feature extractor
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        if freeze_backbone:
            self._freeze()

    def _freeze(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.features(x)
            return nn.AdaptiveAvgPool2d(1)(f).flatten(1)


def build_model(cfg: dict) -> DeepfakeResNet18:
    return DeepfakeResNet18(
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.3),
        freeze_backbone=cfg.get("freeze_backbone_epochs", 0) > 0,
    )
