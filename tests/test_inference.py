"""Tests for predictor utilities that don't require a trained checkpoint."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "src")

def test_tta_returns_8_transforms():
    from deepfake_recognition.data.transforms import get_tta_transforms
    tfs = get_tta_transforms(224)
    assert len(tfs) == 8

def test_gradcam_target_layer_resnet():
    from deepfake_recognition.models.resnet import DeepfakeResNet18
    from deepfake_recognition.utils.gradcam import _resolve_target_layer
    model = DeepfakeResNet18(pretrained=False)
    layer = _resolve_target_layer(model)
    assert layer is not None

def test_gradcam_target_layer_efficientnet():
    from deepfake_recognition.models.efficientnet import DeepfakeEfficientNet
    from deepfake_recognition.utils.gradcam import _resolve_target_layer
    model = DeepfakeEfficientNet(pretrained=False)
    layer = _resolve_target_layer(model)
    assert layer is not None
