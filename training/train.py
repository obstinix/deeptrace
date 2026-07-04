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
    p.add_argument("--resume", action="store_true", help="Resume training from last.pth")
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

    kwargs = {}
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=nw, pin_memory=True, **kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True, **kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True, **kwargs)

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
# Scheduler: linear warmup -> cosine decay
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
    ax.set_title("Confusion matrix - test set")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] confusion matrix -> {path}")


def save_roc_curve(fpr, tpr, auc, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC curve - test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] ROC curve -> {path}")


def save_loss_curves(history, arch):
    out_dir = Path("training/logs") / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "loss_curves.png"
    
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    vl_loss = [h["val_loss"] for h in history]
    tr_acc = [h["train_acc"] for h in history]
    vl_acc = [h["val_acc"] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(epochs, tr_loss, label="Train Loss", marker='o')
    ax1.plot(epochs, vl_loss, label="Val Loss", marker='o')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    
    ax2.plot(epochs, tr_acc, label="Train Acc", marker='o')
    ax2.plot(epochs, vl_acc, label="Val Acc", marker='o')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] loss curves -> {path}")


def check_overfitting_warning(history):
    if len(history) < 5:
        return
    last_5 = history[-5:]
    val_losses = [h["val_loss"] for h in last_5]
    train_losses = [h["train_loss"] for h in last_5]
    
    val_loss_increased = val_losses[-1] > val_losses[0]
    train_loss_decreased = (train_losses[0] - train_losses[-1]) > 0.01
    
    if val_loss_increased and train_loss_decreased:
        print("\n" + "!" * 60)
        print("WARNING: OVERFITTING DETECTED!")
        print(f"Over the last 5 epochs, validation loss increased from {val_losses[0]:.4f} to {val_losses[-1]:.4f}")
        print(f"while training loss decreased from {train_losses[0]:.4f} to {train_losses[-1]:.4f}.")
        print("Consider stopping training or applying stronger regularization.")
        print("!" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# Config normalization and progressive unfreezing helpers
# ──────────────────────────────────────────────────────────────
def scale_scheduler_lr(scheduler, factor):
    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]
    
    # Handle composite/wrapped schedulers like SequentialLR
    if hasattr(scheduler, "_schedulers"):
        for sub_sched in scheduler._schedulers:
            scale_scheduler_lr(sub_sched, factor)


def normalize_config(cfg, arch):
    # Model architecture
    if "model" in cfg:
        if "architecture" not in cfg["model"] and "name" in cfg["model"]:
            cfg["model"]["architecture"] = cfg["model"]["name"]
    
    # Data directories
    if "data" in cfg:
        if "train_dir" not in cfg["data"] and "root_dir" in cfg["data"]:
            root = cfg["data"]["root_dir"]
            cfg["data"]["train_dir"] = str(Path(root) / "train")
            cfg["data"]["val_dir"]   = str(Path(root) / "val")
            cfg["data"]["test_dir"]  = str(Path(root) / "test")
        
        # Image size
        if "image_size" not in cfg["data"] and "img_size" in cfg["data"]:
            cfg["data"]["image_size"] = cfg["data"]["img_size"]
            
        # num_workers
        if "num_workers" not in cfg["data"]:
            cfg["data"]["num_workers"] = 0
            
    # Training parameters
    if "training" in cfg:
        # Batch size
        if "batch_size" not in cfg["training"] and "batch_size" in cfg["data"]:
            cfg["training"]["batch_size"] = cfg["data"]["batch_size"]
            
        # Learning rate
        if "learning_rate" not in cfg["training"] and "lr" in cfg["training"]:
            cfg["training"]["learning_rate"] = cfg["training"]["lr"]

    # Logging parameters
    if "logging" not in cfg:
        cfg["logging"] = {}
    if "log_dir" not in cfg["logging"]:
        cfg["logging"]["log_dir"] = f"logs/{arch}"
    if "eval_report" not in cfg["logging"]:
        cfg["logging"]["eval_report"] = f"logs/{arch}/eval_report.json"
    if "confusion_matrix" not in cfg["logging"]:
        cfg["logging"]["confusion_matrix"] = f"logs/{arch}/confusion_matrix.png"
    if "roc_curve" not in cfg["logging"]:
        cfg["logging"]["roc_curve"] = f"logs/{arch}/roc_curve.png"

    # Augmentation parameters default if missing
    if "augmentation" not in cfg:
        cfg["augmentation"] = {}
    if "horizontal_flip" not in cfg["augmentation"]:
        cfg["augmentation"]["horizontal_flip"] = True
    if "random_crop" not in cfg["augmentation"]:
        cfg["augmentation"]["random_crop"] = True
    if "color_jitter" not in cfg["augmentation"]:
        cfg["augmentation"]["color_jitter"] = {
            "brightness": 0.3,
            "contrast": 0.3,
            "saturation": 0.2,
            "hue": 0.05
        }
    if "random_erasing" not in cfg["augmentation"]:
        cfg["augmentation"]["random_erasing"] = 0.3
    if "normalize" not in cfg["augmentation"]:
        cfg["augmentation"]["normalize"] = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }


def set_backbone_frozen(model, arch, freeze=True):
    if not freeze:
        if hasattr(model, "unfreeze"):
            model.unfreeze()
            print("[train] Backbone unfreezing successful (using model.unfreeze())")
            return
    else:
        if hasattr(model, "_freeze"):
            model._freeze()
            print("[train] Backbone freezing successful (using model._freeze())")
            return
            
    # Fallback/General method:
    arch = arch.lower()
    head_names = []
    if "resnet" in arch:
        head_names = ["head", "fc"]
    elif "efficientnet" in arch:
        head_names = ["head", "classifier"]
    elif "vit" in arch:
        head_names = ["head"]
        
    for name, param in model.named_parameters():
        is_head = any(hn in name for hn in head_names)
        if not is_head:
            param.requires_grad = not freeze
    print(f"[train] Backbone {'frozen' if freeze else 'unfrozen'} via parameter names search.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    
    # Determine architecture — CLI flag overrides config
    arch = args.arch
    if not arch and "model" in cfg:
        arch = cfg["model"].get("architecture") or cfg["model"].get("name")
    if not arch:
        arch = "resnet18"  # fallback default
    
    # Normalise config key structure
    normalize_config(cfg, arch)
    
    cfg["model"]["architecture"] = arch   # normalise for logging

    # Enable MPS for acceleration on Apple Silicon/Mac GPU (fallback to CPU for ViT to avoid MPS freezes)
    if arch == "vit_b16":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[init] device={device}")

    Path(cfg["logging"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logging"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_to_idx = build_loaders(cfg)

    # Build model using custom get_model registry if available, falling back to factory_build
    try:
        from deepfake_recognition.models import get_model, MODEL_REGISTRY
        if arch in MODEL_REGISTRY:
            cfg["model"]["name"] = arch
            model = get_model(cfg["model"])
        else:
            model = factory_build(
                architecture=arch,
                num_classes=cfg["model"]["num_classes"],
                dropout=cfg["model"].get("dropout", 0.5),
            )
    except Exception as e:
        print(f"[init] Fallback to model factory due to: {e}")
        model = factory_build(
            architecture=arch,
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"].get("dropout", 0.5),
        )
    model = model.to(device)

    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)

    print(f"[model] {arch} | params: {count_parameters(model):,}")
    label_smoothing = cfg["training"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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
    last_path      = Path(cfg["logging"]["checkpoint_dir"]) / "last.pth"
    start_epoch    = 1

    freeze_epochs = cfg["model"].get("freeze_backbone_epochs", 0)
    backbone_unfrozen_done = False

    if freeze_epochs > 0:
        print(f"[init] Configuring progressive unfreezing: backbone frozen for first {freeze_epochs} epochs")
        set_backbone_frozen(model, arch, freeze=True)

    if args.resume and last_path.exists():
        print(f"[resume] Loading checkpoint state from {last_path}...")
        checkpoint = torch.load(last_path, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_val_acc = checkpoint["best_val_acc"]
        patience_left = checkpoint["patience_left"]
        history = checkpoint.get("history", [])
        
        # Resume progressive unfreezing state
        if start_epoch > freeze_epochs > 0:
            print("[resume] Resumed epoch is after freeze window. Unfreezing backbone and lowering base LRs.")
            set_backbone_frozen(model, arch, freeze=False)
            backbone_unfrozen_done = True
            scale_scheduler_lr(scheduler, 0.1)
            
        print(f"[resume] Resuming from epoch {start_epoch} | best_val_acc: {best_val_acc:.4f} | patience_left: {patience_left}")

    print(f"\n{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} "
          f"{'ValLoss':>8} {'ValAcc':>7} {'LR':>10}")
    print("-" * 58)

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        if freeze_epochs > 0:
            if epoch <= freeze_epochs:
                print(f"[train] Epoch {epoch}: Backbone is FROZEN (training head only)")
                set_backbone_frozen(model, arch, freeze=True)
            elif not backbone_unfrozen_done:
                print(f"[train] Epoch {epoch}: Unfreezing backbone and lowering learning rate by 10x")
                set_backbone_frozen(model, arch, freeze=False)
                backbone_unfrozen_done = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                scale_scheduler_lr(scheduler, 0.1)

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

            torch.save({
                "epoch": epoch,
                "model_state_dict": state_dict,
            }, ckpt_path)
            print(f"           * new best ({best_val_acc:.4f}) -> saved to {ckpt_path}")
        else:
            patience_left -= 1

        # Save last checkpoint for resume
        scaler_state = scaler.state_dict() if scaler is not None else None
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler_state,
            "best_val_acc": best_val_acc,
            "patience_left": patience_left,
            "history": history,
        }, last_path)

        if patience_left <= 0:
            print(f"[train] early stopping at epoch {epoch}")
            break

    # Save training history
    hist_path = Path(cfg["logging"]["log_dir"]) / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] history -> {hist_path}")

    # ── Final test set evaluation ──────────────────────────────
    print("\n[eval] loading best checkpoint for test evaluation ...")
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
    print(f"[eval] report -> {cfg['logging']['eval_report']}")

    save_confusion_matrix(cm, cfg["logging"]["confusion_matrix"])
    save_roc_curve(fpr, tpr, auc, cfg["logging"]["roc_curve"])
    save_loss_curves(history, arch)
    check_overfitting_warning(history)

    # ── Post-Training Temperature Calibration ──────────────────────────────
    print("\n[calibration] starting post-training temperature calibration ...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from calibrate import calibrate_arch
        val_dir = cfg.get("data", {}).get("val_dir", "data/frames/val")
        if not os.path.exists(val_dir):
            val_dir = "data/frames_face/val"
        arch = cfg["model"]["architecture"]
        calibrate_arch(arch, val_dir, str(ckpt_path), device)
    except Exception as e:
        print(f"[calibration] auto-calibration failed: {e}")

    print(f"\n[train] Training complete - best val acc: {best_val_acc:.4f} | test acc: {acc:.4f} | AUC: {auc:.4f}")
    if acc < 0.90:
        print("[warn] test accuracy below 90% target - consider more epochs or data cleaning")


if __name__ == "__main__":
    main()
