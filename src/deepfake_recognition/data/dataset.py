"""
DeepfakeDataset — universal dataset class for all supported deepfake datasets.

Expected directory structure (after running scripts/download_faceforensics.py):
    root_dir/
        real/    <- real face images (JPEG/PNG)
        fake/    <- deepfake face images (JPEG/PNG)

Label mapping: real=0, fake=1
"""

from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from deepfake_recognition.data.splitter import stratified_split


LABEL_MAP = {"real": 0, "fake": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def _cache_key(root_dir: Path) -> str:
    paths = sorted(
        p for cls_dir in [root_dir / "real", root_dir / "fake"]
        if cls_dir.exists()
        for p in cls_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    digest = hashlib.md5(
        "".join(f"{p}:{p.stat().st_mtime}" for p in paths).encode()
    ).hexdigest()
    return digest


class DeepfakeDataset(Dataset):
    """
    Dataset that loads real/fake face images from a directory.
    Supports label caching, corruption handling, and class weight computation.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform=None,
        max_samples: int | None = None,
        seed: int = 42,
        val_ratio: float = 0.15,
        test_ratio: float = 0.10,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.seed = seed

        samples = self._load_or_scan()

        splits = stratified_split(
            samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        self.samples = splits[split]

        if max_samples:
            self.samples = self.samples[:max_samples]

    def _load_or_scan(self) -> list[tuple[Path, int]]:
        manifest_path = self.root_dir / ".dataset_manifest.json"
        current_key = _cache_key(self.root_dir)

        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
            if data.get("cache_key") == current_key:
                return [(Path(e["path"]), e["label"]) for e in data["samples"]]

        samples = []
        for cls_name, label in LABEL_MAP.items():
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory: {cls_dir}\n"
                    f"Run: python scripts/download_faceforensics.py verify --path {self.root_dir}"
                )
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((p, label))

        with open(manifest_path, "w") as f:
            json.dump({
                "cache_key": current_key,
                "samples": [{"path": str(p), "label": l} for p, l in samples]
            }, f)

        print(f"Scanned {len(samples):,} images → manifest cached")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"WARNING: Corrupted image {path}: {e} — using blank image")
            img = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            img = self.transform(img)

        return img, label, str(path)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for imbalanced datasets."""
        labels = [s[1] for s in self.samples]
        n = len(labels)
        n_real = labels.count(0)
        n_fake = labels.count(1)
        w_real = n / (2 * n_real) if n_real > 0 else 1.0
        w_fake = n / (2 * n_fake) if n_fake > 0 else 1.0
        return torch.tensor([w_real, w_fake], dtype=torch.float32)

    @classmethod
    def from_config(cls, cfg: dict, split: str) -> "DeepfakeDataset":
        from deepfake_recognition.data.transforms import get_train_transforms, get_val_transforms
        transform = get_train_transforms(cfg["img_size"]) if split == "train" \
                    else get_val_transforms(cfg["img_size"])
        return cls(
            root_dir=cfg["root_dir"],
            split=split,
            transform=transform,
            max_samples=cfg.get("max_samples"),
            seed=cfg.get("seed", 42),
            val_ratio=cfg.get("val_ratio", 0.15),
            test_ratio=cfg.get("test_ratio", 0.10),
        )
