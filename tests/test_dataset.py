import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, "src")

@pytest.fixture
def dummy_data(tmp_path):
    for cls in ["real", "fake"]:
        d = tmp_path / cls; d.mkdir()
        for i in range(10):
            Image.new("RGB", (64, 64), color=(i*20, i*10, 0)).save(d / f"{i:03d}.jpg")
    return tmp_path

def test_dataset_length(dummy_data):
    from deepfake_recognition.data.dataset import DeepfakeDataset
    assert len(DeepfakeDataset(dummy_data, split="train")) > 0

def test_dataset_returns_tensor(dummy_data):
    from deepfake_recognition.data.dataset import DeepfakeDataset
    from deepfake_recognition.data.transforms import get_val_transforms
    ds = DeepfakeDataset(dummy_data, split="train", transform=get_val_transforms(64))
    img, label, path = ds[0]
    assert isinstance(img, torch.Tensor) and img.shape[0] == 3
    assert label in (0, 1)

def test_class_weights(dummy_data):
    from deepfake_recognition.data.dataset import DeepfakeDataset
    w = DeepfakeDataset(dummy_data).class_weights()
    assert w.shape == (2,) and (w > 0).all()
