#!/usr/bin/env python3
"""
Train a deepfake detection model.

Usage:
  python training/train.py --config training/configs/resnet18.yaml --data data/frames
  python training/train.py --config training/configs/efficientnet_b3.yaml --data data/frames
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_recognition.data.dataset import DeepfakeDataset
from deepfake_recognition.data.transforms import get_train_transforms, get_val_transforms
from deepfake_recognition.models import get_model
from deepfake_recognition.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data", required=True, help="Path to data/frames directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for quick testing")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override data path from CLI
    cfg["data"]["root_dir"] = args.data
    train_cfg = {**cfg["data"], **cfg["training"], **cfg.get("logging", {})}

    print(f"Config: {args.config}")
    print(f"Data:   {args.data}")
    print(f"Model:  {cfg['model']['name']}")
    print(f"Device: {args.device}")

    # Build datasets
    train_ds = DeepfakeDataset(
        root_dir=args.data, split="train",
        transform=get_train_transforms(cfg["data"]["img_size"]),
        max_samples=args.max_samples,
    )
    val_ds = DeepfakeDataset(
        root_dir=args.data, split="val",
        transform=get_val_transforms(cfg["data"]["img_size"]),
    )

    print(f"Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")
    print(f"Class weights: {train_ds.class_weights()}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"],
        pin_memory=True, persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"] * 2,
        shuffle=False, num_workers=cfg["data"]["num_workers"],
        pin_memory=True, persistent_workers=cfg["data"]["num_workers"] > 0,
    )

    # Build model
    model = get_model(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model params: {n_params:.1f}M total, {n_trainable:.1f}M trainable")

    # Train
    trainer = Trainer(model, train_loader, val_loader, train_cfg, device=args.device)
    best = trainer.fit()

    print(f"\nBest val metrics: {json.dumps({k:v for k,v in best.items() if k!='confusion_matrix'}, indent=2)}")

    # Update AGENT_STATE.json
    state_path = Path("AGENT_STATE.json")
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        state["ml_metrics"] = {
            "best_val_accuracy": best.get("accuracy"),
            "best_val_auc": best.get("auc"),
            "best_val_f1": best.get("f1"),
            "best_epoch": best.get("epoch"),
            "model": cfg["model"]["name"],
            "dataset": args.data,
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        print("Updated AGENT_STATE.json with training metrics.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
