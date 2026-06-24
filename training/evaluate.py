#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on the test split.

Usage:
  python training/evaluate.py \
      --checkpoint checkpoints/resnet18/best.pth \
      --config training/configs/resnet18.yaml \
      --data data/frames
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_recognition.data.dataset import DeepfakeDataset
from deepfake_recognition.data.transforms import get_val_transforms
from deepfake_recognition.models import get_model
from deepfake_recognition.training.metrics import MetricTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(cfg["model"])
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"Checkpoint metrics: {ckpt.get('metrics', {})}")

    ds = DeepfakeDataset(
        root_dir=args.data, split=args.split,
        transform=get_val_transforms(cfg["data"]["img_size"]),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"Evaluating on {args.split}: {len(ds):,} samples")

    tracker = MetricTracker()
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels, paths in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            tracker.update(logits, labels, loss.item())

    metrics = tracker.compute()
    print("\nTest metrics:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  Loss     : {metrics['loss']:.4f}")
    print(f"  Confusion matrix: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
