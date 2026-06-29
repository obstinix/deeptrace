"""
training/merge_kaggle_dataset.py

Merges a subset of the Kaggle deepfake-vs-real-20k dataset into
the local data/frames splits to improve training generalization.
"""
import os
import shutil
import random
from pathlib import Path

KAGGLER_ROOT = "/Users/apple/.cache/kagglehub/datasets/prithivsakthiur/deepfake-vs-real-20k/versions/1/Deep-vs-Real"
TARGET_ROOT  = "data/frames"

SUBSET_COUNT = 1000  # 1000 real and 1000 fake

def merge_split(src_dir: str, target_sub: str, files: list, label: str):
    """Copy files to the target split directory under the given label."""
    dest_dir = Path(TARGET_ROOT) / target_sub / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for f in files:
        src_path = Path(src_dir) / f
        dest_path = dest_dir / f"kaggle_{f}"
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            copied += 1
    print(f"Copied {copied} files to {dest_dir}")

def main():
    if not os.path.exists(KAGGLER_ROOT):
        print(f"Error: Kaggle dataset root not found at {KAGGLER_ROOT}")
        return

    random.seed(42)
    
    # Process Real images
    real_src = Path(KAGGLER_ROOT) / "Real"
    real_files = [f for f in os.listdir(real_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(real_files)
    selected_real = real_files[:SUBSET_COUNT]
    
    # Process Deepfake images
    fake_src = Path(KAGGLER_ROOT) / "Deepfake"
    fake_files = [f for f in os.listdir(fake_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(fake_files)
    selected_fake = fake_files[:SUBSET_COUNT]
    
    # Splits configuration: 80% train, 10% val, 10% test
    n_train = int(SUBSET_COUNT * 0.80)
    n_val   = int(SUBSET_COUNT * 0.10)
    
    # Real splits
    train_real = selected_real[:n_train]
    val_real   = selected_real[n_train:n_train+n_val]
    test_real  = selected_real[n_train+n_val:]
    
    # Fake splits
    train_fake = selected_fake[:n_train]
    val_fake   = selected_fake[n_train:n_train+n_val]
    test_fake  = selected_fake[n_train+n_val:]
    
    print("Starting merge of Kaggle dataset subset (1000 real, 1000 fake)...")
    
    # Copy Real
    merge_split(real_src, "train", train_real, "real")
    merge_split(real_src, "val", val_real, "real")
    merge_split(real_src, "test", test_real, "real")
    
    # Copy Fake
    merge_split(fake_src, "train", train_fake, "fake")
    merge_split(fake_src, "val", val_fake, "fake")
    merge_split(fake_src, "test", test_fake, "fake")
    
    print("\n✓ Merge completed successfully!")

if __name__ == "__main__":
    main()
