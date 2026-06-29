"""
training/fit_ensemble.py

Fit the DeepTrace ensemble meta-classifier on the validation set.

For each trained model, extract its calibrated P(fake) on every image in
data/frames/val. Then fit a logistic regression that takes the three
per-model probabilities as input and predicts the ground truth label.

This learns per-architecture weights that are empirically optimal for
the specific models and training data, rather than hand-tuned defaults.

Usage:
    python training/fit_ensemble.py
    python training/fit_ensemble.py --strategy weighted_average
    python training/fit_ensemble.py --strategy learned --save

Output:
    checkpoints/ensemble/weights.json
    logs/ensemble/fit_report.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deepfake_recognition.utils.model_factory import (
    build_model, SUPPORTED_ARCHITECTURES,
)
from src.deepfake_recognition.utils.calibration import TemperatureScaler
from src.deepfake_recognition.utils.ensemble import EnsembleScorer, DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Probability extraction
# ---------------------------------------------------------------------------

def extract_member_probs(
    arch:       str,
    val_dir:    str,
    ckpt_path:  str,
    temp_path:  str,
    device:     torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model on the full val set and return calibrated P(fake) + labels.
    Returns (probs, labels) — both shape (N,).
    """
    val_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(val_dir, transform=val_tf)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=False)  # macOS compatibility
    class_to_idx = dataset.class_to_idx
    fake_idx     = class_to_idx.get("fake", 0)

    model = build_model(arch, num_classes=2, dropout=0.0)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # Load calibrator if present
    calibrator = None
    if Path(temp_path).exists():
        calibrator = TemperatureScaler.load(temp_path)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            if calibrator is not None:
                probs = calibrator.calibrate(logits)
            else:
                probs = torch.softmax(logits, dim=1)
            all_probs.append(probs[:, fake_idx].cpu().numpy())
            all_labels.append(lbls.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def auc_roc(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, probs))


def accuracy(probs: np.ndarray, labels: np.ndarray,
             threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(int)
    return float((preds == labels).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DeepTrace ensemble")
    p.add_argument("--strategy", default="learned",
                   choices=["weighted_average", "learned"],
                   help="Ensemble strategy to fit (default: learned)")
    p.add_argument("--val-dir",  default=None,
                   help="Override val directory")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--save",     action="store_true", default=True)
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"[ensemble] device: {device}")
    print(f"[ensemble] strategy: {args.strategy}")

    # Resolve val dir
    val_dir = args.val_dir
    if val_dir is None:
        for candidate in ("data/frames_face/val", "data/frames/val"):
            if Path(candidate).exists():
                val_dir = candidate
                break
    if val_dir is None or not Path(val_dir).exists():
        print("[ensemble] ERROR: val directory not found. "
              "Pass --val-dir explicitly.")
        sys.exit(1)
    print(f"[ensemble] val_dir: {val_dir}")

    # Extract per-model calibrated probabilities
    member_cols   = []   # each element: (N,) float array of P(fake)
    feature_order = []
    labels_ref    = None

    for arch, ckpt in SUPPORTED_ARCHITECTURES.items():
        temp_path = str(Path(ckpt).parent / "temperature.json")
        if not Path(ckpt).exists():
            print(f"[ensemble] SKIP {arch} — checkpoint not found: {ckpt}")
            continue

        print(f"\n[ensemble] extracting probs: {arch} …")
        probs, labels = extract_member_probs(
            arch, val_dir, ckpt, temp_path, device
        )

        if labels_ref is None:
            labels_ref = labels
        else:
            assert np.array_equal(labels_ref, labels), \
                f"Label mismatch between {feature_order[0]} and {arch}"

        member_cols.append(probs)
        feature_order.append(arch)

        ind_auc = auc_roc(probs, labels)
        ind_acc = accuracy(probs, labels)
        print(f"  {arch}: AUC={ind_auc:.4f}  Acc={ind_acc:.4f}")

    if len(member_cols) < 2:
        print("[ensemble] ERROR: need at least 2 loaded models to fit ensemble")
        sys.exit(1)

    # Stack into matrix: (N, n_models)
    X      = np.stack(member_cols, axis=1)   # (N, n_models)
    labels = labels_ref

    print(f"\n[ensemble] matrix shape: {X.shape}  |  "
          f"fake fraction: {labels.mean():.3f}")

    scorer = EnsembleScorer(
        strategy="weighted_average",   # init with defaults; fit() overrides
        weights_path="checkpoints/ensemble/weights.json",
    )

    if args.strategy == "weighted_average":
        # Derive weights from individual AUC scores
        aucs    = [auc_roc(X[:, i], labels) for i in range(X.shape[1])]
        raw_w   = {arch: max(0.0, auc - 0.5) for arch, auc in zip(feature_order, aucs)}
        scorer.fit_weighted(raw_w, save=args.save)
        print(f"\n[ensemble] derived weights: {scorer.active_weights}")

    elif args.strategy == "learned":
        # Labels: 1 = fake, 0 = real
        # ImageFolder: fake_idx might be 0 — align to convention here
        fake_idx   = 0   # must match class_to_idx["fake"] from ImageFolder
        # If fake_idx == 0, then labels==0 means fake, labels==1 means real
        # Logistic regression needs fake=1
        bin_labels = (labels == fake_idx).astype(int)
        scorer.fit_learned(X, bin_labels, feature_order, save=args.save)

    # Evaluate ensemble on the same val set
    ens_probs = np.array([
        scorer.score({arch: float(X[i, j])
                      for j, arch in enumerate(feature_order)})["ensemble_fake_prob"]
        for i in range(len(labels))
    ])

    ens_labels = (labels == 0).astype(int)   # fake_idx=0 → fake=1
    ens_auc    = auc_roc(ens_probs, ens_labels)
    ens_acc    = accuracy(ens_probs, ens_labels)

    print(f"\n[ensemble] ensemble AUC: {ens_auc:.4f}")
    print(f"[ensemble] ensemble Acc: {ens_acc:.4f}")

    # Improvement over best individual model
    best_ind_auc = max(auc_roc(X[:, i], ens_labels) for i in range(X.shape[1]))
    print(f"[ensemble] best individual AUC: {best_ind_auc:.4f}")
    print(f"[ensemble] improvement: {ens_auc - best_ind_auc:+.4f}")

    # Save fit report
    log_dir = Path("logs/ensemble")
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "strategy":          args.strategy,
        "feature_order":     feature_order,
        "n_val_samples":     int(len(labels)),
        "ensemble_auc":      round(ens_auc, 6),
        "ensemble_acc":      round(ens_acc, 6),
        "best_individual_auc": round(best_ind_auc, 6),
        "auc_improvement":   round(ens_auc - best_ind_auc, 6),
        "individual_metrics": {
            arch: {
                "auc": round(auc_roc(X[:, i], ens_labels), 6),
                "acc": round(accuracy(X[:, i], ens_labels), 6),
            }
            for i, arch in enumerate(feature_order)
        },
    }
    if args.strategy == "weighted_average":
        report["weights"] = scorer.active_weights

    report_path = log_dir / "fit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ensemble] fit report → {report_path}")

    print(f"\n✓ Ensemble fitted ({args.strategy}) — "
          f"AUC {ens_auc:.4f}  Acc {ens_acc:.4f}")


if __name__ == "__main__":
    main()
