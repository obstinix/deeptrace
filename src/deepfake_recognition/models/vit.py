"""ViT-Base/16 fine-tuned for binary deepfake classification."""
from __future__ import annotations
import torch
import torch.nn as nn
import timm


class DeepfakeViT(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3,
                 freeze_backbone: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 768 for ViT-B/16
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
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


def build_model(cfg: dict) -> DeepfakeViT:
    return DeepfakeViT(
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.3),
        freeze_backbone=cfg.get("freeze_backbone_epochs", 0) > 0,
    )
