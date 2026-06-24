import sys

sys.path.insert(0, "src")
import torch
from PIL import Image


def test_val_shape():
    from deepfake_recognition.data.transforms import get_val_transforms
    assert get_val_transforms(224)(Image.new("RGB",(400,300))).shape == (3,224,224)

def test_val_deterministic():
    from deepfake_recognition.data.transforms import get_val_transforms
    tf = get_val_transforms(224); img = Image.new("RGB",(400,300),(100,150,200))
    assert torch.allclose(tf(img), tf(img))

def test_tta_multiple():
    from deepfake_recognition.data.transforms import get_tta_transforms
    tfs = get_tta_transforms(224)
    assert len(tfs) >= 2
    for tf in tfs: assert tf(Image.new("RGB",(300,300))).shape == (3,224,224)
