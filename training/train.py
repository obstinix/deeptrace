"""
training/train.py
DeepTrace — ResNet-18 fine-tune on binary deepfake classification
"""
import argparse
import json
import os
import time
import sys
from pathlib import Path

# Add src to path to resolve local modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

# Import the custom model to ensure state_dict compatibility with Predictor
from deepfake_recognition.models.resnet import DeepfakeResNet18
from deepfake_recognition.utils.model_factory import (
    build_model as factory_build,
    count_parameters,
)


# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DeepTrace ResNet-18 training")
    p.add_argument("--config", default="training/configs/resnet18.yaml")
    p.add_argument("--data",   default=None, help="Override data root")
    p.add_argument(
        "--arch",
        default=None,
        help="Override architecture (resnet18 | efficientnet_b0). Default: read from config.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────
def build_transforms(cfg, split="train"):
    size = cfg["data"]["image_size"]
    norm = T.Normalize(
        mean=cfg["augmentation"]["normalize"]["mean"],
        std=cfg["augmentation"]["normalize"]["std"],
    )
    if split == "train":
        aug = cfg["augmentation"]
        ops = [T.Resize(int(size * 1.14)), T.RandomCrop(size)]
        if aug.get("horizontal_flip"):
            ops.append(T.RandomHorizontalFlip())
        cj = aug.get("color_jitter", {})
        if cj:
            ops.append(T.ColorJitter(**{k: v for k, v in cj.items()}))
        ops += [T.ToTensor(), norm]
        p_erase = aug.get("random_erasing", 0)
        if p_erase:
            ops.append(T.RandomErasing(p=p_erase))
        return T.Compose(ops)
    else:
        return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size),
                          T.ToTensor(), norm])


# ──────────────────────────────────────────────────────────────
# Dataset + balanced sampler
# ──────────────────────────────────────────────────────────────
def build_loaders(cfg):
    train_tf = build_transforms(cfg, "train")
    val_tf   = build_transforms(cfg, "val")

    train_ds = ImageFolder(cfg["data"]["train_dir"], transform=train_tf)
    val_ds   = ImageFolder(cfg["data"]["val_dir"],   transform=val_tf)
    test_ds  = ImageFolder(cfg["data"]["test_dir"],  transform=val_tf)

    # Balanced sampler — handles class imbalance without manual weighting
    counts  = np.bincount([s[1] for s in train_ds.samples])
    weights = 1.0 / counts[[s[1] for s in train_ds.samples]]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    nw = cfg["data"]["num_workers"]
    bs = cfg["training"]["batch_size"]

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    print(f"[data] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"[data] class map: {train_ds.class_to_idx}")  # {'fake': 0, 'real': 1} or flipped
    return train_loader, val_loader, test_loader, train_ds.class_to_idx


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────
def build_model(cfg, device):
    dropout  = cfg["model"].get("dropout", 0.5)
    # Instantiate custom DeepfakeResNet18 instead of standard resnet18
    model = DeepfakeResNet18(
        pretrained=cfg["model"]["pretrained"],
        dropout=dropout,
        freeze_backbone=False
    )
    return model.to(device)


# ──────────────────────────────────────────────────────────────
# Scheduler: linear warmup → cosine decay
# ──────────────────────────────────────────────────────────────
def build_scheduler(optimizer, cfg):
    warmup = cfg["training"].get("warmup_epochs", 3)
    total  = cfg["training"]["epochs"]
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                        milestones=[warmup])


# ──────────────────────────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, device,
              train=True, grad_accum_steps=1, grad_clip=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad if train else torch.no_grad
    use_autocast = (device.type == "cuda" and scaler is not None)

    if train and optimizer:
        optimizer.zero_grad()

    with ctx():
        for step, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(enabled=use_autocast):
                logits = model(imgs)
                loss   = criterion(logits, labels)
                if train:
                    loss = loss / grad_accum_steps
            if train:
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Step only every N batches, or at the end of the epoch
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                    if grad_clip and scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip
                        )
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip
                            )
                        optimizer.step()
                    optimizer.zero_grad()

            loss_item = loss.item()
            if train:
                loss_item = loss_item * grad_accum_steps
            total_loss += loss_item * imgs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────
# Final evaluation on test set
# ──────────────────────────────────────────────────────────────
def evaluate(model, loader, device, class_to_idx):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    # Determine which index is "fake" (for ROC — positive class)
    fake_idx = class_to_idx.get("fake", 0)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, fake_idx].cpu().numpy())

    acc     = accuracy_score(all_labels, all_preds)
    
    # Map positive class (fake) to 1 and negative class (real) to 0 for AUC calculation
    y_true_binary = [1 if l == fake_idx else 0 for l in all_labels]
    auc     = roc_auc_score(y_true_binary, all_probs)
    
    # Resolve target names dynamically from class_to_idx to match correct label mapping
    target_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    report  = classification_report(all_labels, all_preds,
                                    target_names=target_names, output_dict=True)
    cm      = confusion_matrix(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(y_true_binary, all_probs)
    return acc, auc, report, cm, fpr, tpr


# ──────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────
def save_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["real", "fake"]); ax.set_yticklabels(["real", "fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix — test set")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] confusion matrix → {path}")


def save_roc_curve(fpr, tpr, auc, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC curve — test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] ROC curve → {path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    
    # Enable MPS for acceleration on Apple Silicon/Mac GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[init] device={device}")

    Path(cfg["logging"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logging"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_to_idx = build_loaders(cfg)
    
    # Determine architecture — CLI flag overrides config
    arch = args.arch or cfg["model"]["architecture"]
    cfg["model"]["architecture"] = arch   # normalise for logging

    model = factory_build(
        architecture=arch,
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"].get("dropout", 0.5),
    ).to(device)

    if arch == "vit_b16":
        # Freeze all layers except block 11, norm, and head to support small checkpoint (<30MB)
        for name, param in model.named_parameters():
            if not ("blocks.11." in name or "norm." in name or "head." in name):
                param.requires_grad = False

    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)

    print(f"[model] {arch} | params: {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),
                      lr=cfg["training"]["learning_rate"],
                      weight_decay=cfg["training"]["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg)
    
    # GradScaler is only supported/needed for CUDA
    use_amp = cfg["training"].get("mixed_precision") and device.type == "cuda"
    scaler    = GradScaler() if use_amp else None

    accum = cfg["training"].get("gradient_accumulation_steps", 1)
    clip  = cfg["training"].get("gradient_clip", None)

    best_val_acc   = 0.0
    patience_left  = cfg["training"].get("early_stopping_patience", 8)
    history        = []
    ckpt_path      = Path(cfg["logging"]["checkpoint_dir"]) / "best.pth"

    print(f"\n{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} "
          f"{'ValLoss':>8} {'ValAcc':>7} {'LR':>10}")
    print("─" * 58)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    scaler, device, train=True, grad_accum_steps=accum, grad_clip=clip)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, None,
                                    None, device, train=False, grad_accum_steps=1, grad_clip=None)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch:>5}  {tr_loss:>9.4f}  {tr_acc:>8.4f}  "
              f"{vl_loss:>7.4f}  {vl_acc:>6.4f}  {lr:>10.2e}  "
              f"({elapsed:.0f}s)")

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc, "lr": lr,
        })

        if vl_acc > best_val_acc:
            best_val_acc  = vl_acc
            patience_left = cfg["training"]["early_stopping_patience"]
            # Save dict with model_state_dict to match Predictor.from_checkpoint
            state_dict = model.state_dict()
            if arch == "vit_b16":
                # Save only blocks.11, norm, and head weights to keep checkpoint size < 30MB
                state_dict = {k: v for k, v in state_dict.items() if "blocks.11." in k or "norm." in k or "head." in k}

            torch.save({
                "epoch": epoch,
                "model_state_dict": state_dict,
            }, ckpt_path)
            print(f"           ↑ new best ({best_val_acc:.4f}) → saved to {ckpt_path}")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"[train] early stopping at epoch {epoch}")
                break

    # Save training history
    hist_path = Path(cfg["logging"]["log_dir"]) / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] history → {hist_path}")

    # ── Final test set evaluation ──────────────────────────────
    print("\n[eval] loading best checkpoint for test evaluation …")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
        
    acc, auc, report, cm, fpr, tpr = evaluate(model, test_loader, device, class_to_idx)

    print(f"\n[eval] test accuracy : {acc:.4f}")
    print(f"[eval] AUC-ROC       : {auc:.4f}")
    print(f"[eval] f1-fake       : {report['fake']['f1-score']:.4f}")
    print(f"[eval] f1-real       : {report['real']['f1-score']:.4f}")

    eval_report = {
        "architecture": cfg["model"]["architecture"],
        "dataset": "Celeb-DF v2",
        "test_accuracy": round(acc, 6),
        "auc_roc": round(auc, 6),
        "precision_fake": round(report["fake"]["precision"], 6),
        "recall_fake": round(report["fake"]["recall"], 6),
        "f1_fake": round(report["fake"]["f1-score"], 6),
        "precision_real": round(report["real"]["precision"], 6),
        "recall_real": round(report["real"]["recall"], 6),
        "f1_real": round(report["real"]["f1-score"], 6),
        "best_val_acc": round(best_val_acc, 6),
        "epochs_trained": len(history),
        "checkpoint": str(ckpt_path),
    }
    with open(cfg["logging"]["eval_report"], "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"[eval] report → {cfg['logging']['eval_report']}")

    save_confusion_matrix(cm, cfg["logging"]["confusion_matrix"])
    save_roc_curve(fpr, tpr, auc, cfg["logging"]["roc_curve"])

    # ── Post-Training Temperature Calibration ──────────────────────────────
    print("\n[calibration] starting post-training temperature calibration …")
    try:
        from training.calibrate import calibrate_arch
        val_dir = cfg.get("data", {}).get("val_dir", "data/frames/val")
        if not os.path.exists(val_dir):
            val_dir = "data/frames_face/val"
        arch = cfg["model"]["architecture"]
        calibrate_arch(arch, val_dir, str(ckpt_path), device)
    except Exception as e:
        print(f"[calibration] auto-calibration failed: {e}")

    print(f"\n✓ Training complete — best val acc: {best_val_acc:.4f} | test acc: {acc:.4f} | AUC: {auc:.4f}")
    if acc < 0.90:
        print("[warn] test accuracy below 90% target — consider more epochs or data cleaning")


if __name__ == "__main__":
    main()
