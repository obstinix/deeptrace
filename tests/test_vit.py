import sys
import pytest
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, "src")

def test_vit_b16_creation():
    from deepfake_recognition.utils.model_factory import build_model, SUPPORTED_ARCHITECTURES
    assert "vit_b16" in SUPPORTED_ARCHITECTURES
    
    # Build vit_b16 (untrained)
    model = build_model("vit_b16", num_classes=2)
    assert model is not None
    
    # Forward check
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 2)

def test_supports_gradcam():
    from deepfake_recognition.utils.model_factory import supports_gradcam
    assert not supports_gradcam("vit_b16")
    assert supports_gradcam("resnet18")
    assert supports_gradcam("efficientnet_b0")

def test_attention_rollout():
    from deepfake_recognition.utils.model_factory import build_model
    from deepfake_recognition.utils.attention_rollout import AttentionRollout
    
    model = build_model("vit_b16").eval()
    x = torch.randn(1, 3, 224, 224)
    
    rollout = AttentionRollout(model)
    mask = rollout(x)
    
    # Mask checks
    assert mask.shape == (14, 14)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    
    # Overlay checks
    img = Image.fromarray(np.zeros((224, 224, 3), dtype="uint8"))
    overlay = AttentionRollout.overlay(img, mask)
    assert overlay is not None
    assert overlay.size == (224, 224)

def test_gradcam_guard_on_vit():
    from deepfake_recognition.utils.model_factory import build_model
    from deepfake_recognition.utils.gradcam import GradCAM
    
    model = build_model("vit_b16")
    with pytest.raises(ValueError) as excinfo:
        GradCAM(model, target_layer=None)
    assert "ViT-B/16 uses AttentionRollout" in str(excinfo.value)
