import sys

import torch

sys.path.insert(0, "src")

def test_resnet18_shape():
    from deepfake_recognition.models.resnet import DeepfakeResNet18
    assert DeepfakeResNet18(pretrained=False)(torch.randn(2,3,224,224)).shape == (2,2)

def test_resnet18_freeze_unfreeze():
    from deepfake_recognition.models.resnet import DeepfakeResNet18
    m = DeepfakeResNet18(pretrained=False, freeze_backbone=True)
    assert not any(p.requires_grad for p in m.features.parameters())
    m.unfreeze()
    assert all(p.requires_grad for p in m.features.parameters())

def test_efficientnet_shape():
    from deepfake_recognition.models.efficientnet import DeepfakeEfficientNet
    assert DeepfakeEfficientNet(pretrained=False)(torch.randn(2,3,300,300)).shape == (2,2)

def test_model_registry():
    from deepfake_recognition.models import get_model
    m = get_model({"name":"resnet18","pretrained":False,"dropout":0.3,"freeze_backbone_epochs":0})
    assert m is not None

def test_unknown_model_raises():
    import pytest

    from deepfake_recognition.models import get_model
    with pytest.raises(ValueError): get_model({"name":"nonexistent"})
