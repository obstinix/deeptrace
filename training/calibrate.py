"""
training/calibrate.py

Standalone calibration script for DeepTrace models.
Run after training — fits temperature T on the validation set.

Usage:
    # Calibrate all trained models
    python training/calibrate.py --all

    # Calibrate a specific architecture
    python training/calibrate.py --arch resnet18 --config training/configs/resnet18.yaml

    # Calibrate with a custom validation set
    python training/calibrate.py --arch vit_b16 --val-dir data/frames_face/val

Output per model:
    checkpoints/{arch}/temperature.json
    logs/{arch}/calibration_report.json
    logs/{arch}/reliability_diagram.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deepfake_recognition.utils.model_factory import (
    build_model,
    SUPPORTED_ARCHITECTURES,
)
from src.deepfake_recognition.utils.calibration import (
    TemperatureScaler,
    expected_calibration_error,
)


# ---------------------------------------------------------------------------
# Logit extraction
# ---------------------------------------------------------------------------

def extract_logits(
    arch:       str,
    val_dir:    str,
    ckpt:       str,
    device:     torch.device,
    batch_size: int = 64,
):
    """
    Run the model on the full validation set and return raw logits + labels.

    Returns:
        (logits, labels, class_to_idx)
        logits:  (N, 2) float32 numpy array — raw pre-softmax
        labels:  (N,)   int numpy array
        class_to_idx: {"fake": 0, "real": 1} or reverse
    """
    val_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    dataset      = ImageFolder(val_dir, transform=val_tf)
    loader       = DataLoader(dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    class_to_idx = dataset.class_to_idx
    print(f"[calibrate] {arch}: {len(dataset)} val samples | "
          f"class_to_idx={class_to_idx}")

    model = build_model(arch, num_classes=2, dropout=0.0)   # dropout=0 at eval
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            all_logits.append(out.cpu().numpy())
            all_labels.append(lbls.numpy())

    logits = np.vstack(all_logits)   # (N, 2)
    labels = np.concatenate(all_labels)
    return logits, labels, class_to_idx


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def save_reliability_diagram(
    pre_bins:  list,
    post_bins: list,
    pre_ece:   float,
    post_ece:  float,
    arch:      str,
    path:      str,
) -> None:
    """
    Plot a reliability diagram showing calibration before and after.
    A perfectly calibrated model lies on the diagonal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, bins, ece, label, color in [
        (axes[0], pre_bins,  pre_ece,  "Before (uncalibrated)", "#ff6b6b"),
        (axes[1], post_bins, post_ece, "After  (T-scaled)",     "#51cf66"),
    ]:
        if not bins:
            ax.text(0.5, 0.5, "No bin data", ha="center", va="center")
            continue

        midpoints  = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
        accuracies = [b["mean_acc"]  for b in bins]
        confs      = [b["mean_conf"] for b in bins]
        widths     = [b["bin_hi"] - b["bin_lo"] for b in bins]

        # Confidence bars
        ax.bar(midpoints, accuracies, width=widths,
               alpha=0.7, color=color, label="Accuracy", align="center",
               edgecolor="white", linewidth=0.5)

        # Gap overlay (overconfidence = bar above diagonal)
        ax.bar(midpoints, confs, width=widths,
               alpha=0.2, color="grey", label="Confidence", align="center")

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{arch} — {label}\nECE = {ece:.4f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[calibrate] reliability diagram -> {path}")


# ---------------------------------------------------------------------------
# Single-arch calibration
# ---------------------------------------------------------------------------

def calibrate_arch(
    arch:    str,
    val_dir: str,
    ckpt:    str,
    device:  torch.device,
) -> None:
    print(f"\n{'-'*60}")
    print(f"[calibrate] architecture: {arch}")
    print(f"[calibrate] checkpoint:   {ckpt}")
    print(f"[calibrate] val_dir:      {val_dir}")

    if not Path(ckpt).exists():
        print(f"[calibrate] SKIP — checkpoint not found: {ckpt}")
        return
    if not Path(val_dir).exists():
        print(f"[calibrate] SKIP — val_dir not found: {val_dir}")
        return

    # Extract raw logits from validation set
    logits, labels, class_to_idx = extract_logits(arch, val_dir, ckpt, device)
    fake_idx = class_to_idx.get("fake", 0)

    # Fit temperature
    scaler = TemperatureScaler()
    scaler.fit(logits, labels, fake_class_idx=fake_idx, verbose=True)

    # Save temperature file
    temp_path = Path(ckpt).parent / "temperature.json"
    scaler.save(str(temp_path))

    # Save calibration report
    log_dir   = Path("logs") / arch
    log_dir.mkdir(parents=True, exist_ok=True)

    report_path = log_dir / "calibration_report.json"
    report      = {
        "architecture": arch,
        "checkpoint":   ckpt,
        "temperature":  scaler.temperature,
        **scaler._fit_meta,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[calibrate] report -> {report_path}")

    # Save reliability diagram
    diagram_path = log_dir / "reliability_diagram.png"
    save_reliability_diagram(
        pre_bins  = scaler._fit_meta.get("pre_bins",  []),
        post_bins = scaler._fit_meta.get("post_bins", []),
        pre_ece   = scaler._fit_meta.get("pre_ece",   0.0),
        post_ece  = scaler._fit_meta.get("post_ece",  0.0),
        arch      = arch,
        path      = str(diagram_path),
    )

    print(f"\n[calibrate] [OK] {arch}")
    print(f"            T              = {scaler.temperature:.4f}")
    print(f"            ECE before     = {scaler._fit_meta.get('pre_ece',  0):.5f}")
    print(f"            ECE after      = {scaler._fit_meta.get('post_ece', 0):.5f}")
    print(f"            ECE reduction  = {scaler._fit_meta.get('ece_improvement', 0):+.5f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DeepTrace temperature calibration")
    p.add_argument("--arch",    default=None,
                   help="Single architecture to calibrate")
    p.add_argument("--config",  default=None,
                   help="Training config YAML (used to find val_dir)")
    p.add_argument("--val-dir", default=None,
                   help="Override val directory path")
    p.add_argument("--all",     action="store_true",
                   help="Calibrate all architectures in SUPPORTED_ARCHITECTURES")
    p.add_argument("--device",  default="cuda",
                   help="cpu | cuda (default: cuda)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device == "cuda" else "cpu")
    print(f"[calibrate] device: {device}")

    if args.all:
        # Default val_dir — update if your face-crop dataset path differs
        default_val = "data/frames/val"
        if not Path(default_val).exists():
            default_val = "data/frames_face/val"   # face-crop variant

        for arch, ckpt in SUPPORTED_ARCHITECTURES.items():
            # Try to find a config with a val_dir
            config_path = Path(f"training/configs/{arch.replace('_', '')}.yaml")
            if not config_path.exists():
                config_path = Path(f"training/configs/{arch}.yaml")
            val_dir = default_val
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path) as f:
                        cfg = yaml.safe_load(f)
                    val_dir = cfg.get("data", {}).get("val_dir", default_val)
                except ImportError:
                    pass

            calibrate_arch(arch, val_dir, ckpt, device)
        return

    if not args.arch:
        print("[calibrate] ERROR: provide --arch or --all")
        sys.exit(1)

    # Resolve val_dir
    val_dir = args.val_dir
    if val_dir is None and args.config:
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            val_dir = cfg.get("data", {}).get("val_dir", "data/frames/val")
        except ImportError:
            val_dir = "data/frames/val"
    if val_dir is None:
        val_dir = "data/frames/val"
        if not Path(val_dir).exists():
            val_dir = "data/frames_face/val"

    ckpt = SUPPORTED_ARCHITECTURES.get(args.arch.lower())
    if not ckpt:
        print(f"[calibrate] ERROR: unknown architecture '{args.arch}'")
        sys.exit(1)

    calibrate_arch(args.arch.lower(), val_dir, ckpt, device)


if __name__ == "__main__":
    main()
