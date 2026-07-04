"""
src/deepfake_recognition/utils/model_factory.py
Build any supported DeepTrace architecture with a unified interface.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

# Registry of supported architectures and their checkpoint sub-dirs
SUPPORTED_ARCHITECTURES = {
    "resnet18":        "checkpoints/resnet18/best.pth",
    "efficientnet_b0": "checkpoints/efficientnet_b0/best.pth",
    "efficientnet_b3": "checkpoints/efficientnet_b3/best.pth",
    "vit_b16":         "checkpoints/vit_b16/best.pth",
    "vit_base":        "checkpoints/vit_base/best.pth",
}

# Input image sizes required per architecture
IMAGE_SIZES = {
    "resnet18":        224,
    "efficientnet_b0": 224,
    "efficientnet_b3": 300,
    "vit_b16":         224,
    "vit_base":        224,
}


def build_model(architecture: str, num_classes: int = 2, dropout: float = 0.5) -> nn.Module:
    """
    Build and return an untrained model for the given architecture.
    Call .load_state_dict() on the result to load a checkpoint.
    """
    arch = architecture.lower()

    try:
        from deepfake_recognition.models import get_model, MODEL_REGISTRY
        if arch in MODEL_REGISTRY:
            return get_model({
                "name": arch,
                "pretrained": True,
                "num_classes": num_classes,
                "dropout": dropout,
                "freeze_backbone_epochs": 0
            })
    except Exception as e:
        print(f"[model_factory] get_model import failed, fallback to native build: {e}")

    if arch == "resnet18":
        try:
            from deepfake_recognition.models.resnet import DeepfakeResNet18
            return DeepfakeResNet18(pretrained=True, dropout=dropout, freeze_backbone=False)
        except Exception:
            weights = tv_models.ResNet18_Weights.DEFAULT
            model   = tv_models.resnet18(weights=weights)
            in_feats = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_feats, num_classes),
            )
            return model

    if arch == "efficientnet_b0":
        if _TIMM_AVAILABLE:
            # timm gives access to pretrained EfficientNet weights
            model = timm.create_model(
                "efficientnet_b0",
                pretrained=True,
                num_classes=num_classes,
                drop_rate=dropout,
            )
            return model
        else:
            # Fallback: torchvision EfficientNet-B0 (available since 0.13)
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT
            model   = tv_models.efficientnet_b0(weights=weights)
            in_feats = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_feats, num_classes),
            )
            return model

    if arch == "vit_b16":
        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for ViT-B/16. Install with: pip install timm"
            )
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes,
            # drop_rate applies to the MLP blocks, attn_drop_rate to attention weights
            drop_rate=dropout,
            attn_drop_rate=0.0,    # keep attention clean for rollout visualisation
        )
        return model

    raise ValueError(
        f"Unknown architecture: '{architecture}'. "
        f"Supported: {list(SUPPORTED_ARCHITECTURES.keys())}"
    )


def get_gradcam_target_layer(model: nn.Module, architecture: str):
    """
    Return the layer to hook for Grad-CAM for the given architecture.
    Must be the last convolutional layer before the classifier head.
    """
    arch = architecture.lower()

    if "vit" in arch:
        # ViT uses AttentionRollout, not Grad-CAM.
        # Returning None signals the API to use the attention path.
        return None

    if arch == "resnet18":
        if hasattr(model, "features"):
            return model.features[-1][-1]        # DeepfakeResNet18 features
        return model.layer4[-1]                  # BasicBlock at end of layer4

    if "efficientnet" in arch:
        if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
            return model.backbone.blocks[-1][-1]
        elif _TIMM_AVAILABLE and hasattr(model, "blocks"):
            return model.blocks[-1][-1]          # last MBConv block
        else:
            return model.features[-1]            # torchvision: last Conv2dNormActivation

    raise ValueError(f"No Grad-CAM target defined for '{architecture}'")


def supports_gradcam(architecture: str) -> bool:
    """Return True if this architecture uses Grad-CAM for explainability.
    ViT uses AttentionRollout instead."""
    return "vit" not in architecture.lower()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
