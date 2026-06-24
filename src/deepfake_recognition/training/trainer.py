"""
Main training loop with mixed precision, gradient clipping, and WandB logging.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_recognition.training.callbacks import EarlyStopping, ModelCheckpoint
from deepfake_recognition.training.metrics import MetricTracker


class Trainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader,
                 cfg: dict, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else \
                     "mps"  if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        # Loss: class-weighted + label smoothing
        train_ds = train_loader.dataset
        if cfg.get("use_class_weights") and hasattr(train_ds, "class_weights"):
            weights = train_ds.class_weights().to(device)
        else:
            weights = None
        self.criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=cfg.get("label_smoothing", 0.05)
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4)
        )

        # Scheduler: cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg["epochs"], eta_min=1e-6
        )

        # Mixed precision
        self.use_amp = cfg.get("mixed_precision", True) and device == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # Callbacks
        self.early_stop = EarlyStopping(
            patience=cfg.get("early_stopping_patience", 8)
        )
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints/"),
            save_top_k=cfg.get("save_top_k", 3),
        )

        # WandB (optional)
        self.use_wandb = cfg.get("use_wandb", False)
        if self.use_wandb:
            import wandb
            wandb.init(project=cfg.get("project", "deepfake-recognition"), config=cfg)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        tracker = MetricTracker()
        freeze_epochs = self.cfg.get("freeze_backbone_epochs", 0)
        if epoch == freeze_epochs and hasattr(self.model, "unfreeze"):
            print(f"  Epoch {epoch}: unfreezing backbone")
            self.model.unfreeze()
            # Re-init optimizer with all params now unfrozen
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg["lr"] * 0.1,   # lower LR after unfreeze
                weight_decay=self.cfg.get("weight_decay", 1e-4)
            )

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}", leave=False)
        for images, labels, _ in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device,
                                     dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.get("gradient_clip", 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tracker.update(logits, labels, loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return tracker.compute()

    @torch.no_grad()
    def val_epoch(self) -> dict:
        self.model.eval()
        tracker = MetricTracker()
        for images, labels, _ in tqdm(self.val_loader, desc="  Val", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.amp.autocast(device_type=self.device,
                                     dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            tracker.update(logits, labels, loss.item())
        return tracker.compute()

    def fit(self) -> dict:
        best_metrics = {}
        print(f"\nTraining on {self.device} for {self.cfg['epochs']} epochs\n")

        for epoch in range(1, self.cfg["epochs"] + 1):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m = self.val_epoch()
            self.scheduler.step()
            elapsed = time.time() - t0

            print(
                f"E{epoch:03d} | "
                f"train_loss={train_m['loss']:.4f} acc={train_m['accuracy']:.3f} | "
                f"val_loss={val_m['loss']:.4f} acc={val_m['accuracy']:.3f} "
                f"auc={val_m['auc']:.4f} f1={val_m['f1']:.3f} | "
                f"{elapsed:.0f}s"
            )

            if self.use_wandb:
                import wandb
                wandb.log({"epoch": epoch, "train": train_m, "val": val_m,
                           "lr": self.optimizer.param_groups[0]["lr"]})

            self.checkpoint.save(self.model, self.optimizer, epoch, val_m)

            if val_m["auc"] > best_metrics.get("auc", 0):
                best_metrics = {**val_m, "epoch": epoch}

            if self.early_stop.step(val_m):
                print(f"Early stopping at epoch {epoch}")
                break

        if self.use_wandb:
            import wandb
            wandb.finish()

        return best_metrics
