#!/usr/bin/env python3
"""
verify_training.py — quick leakage/overfitting sanity checks for a DeepTrace run.

Run from the repo root, e.g.:
    python scripts/verify_training.py --arch resnet18 --data-dir data/frames

Checks:
  1. Exact-duplicate images across train/val/test (file-hash based)
  2. Per-epoch train/val accuracy trajectory + gap, flagging suspicious jumps
Doesn't need a GPU and doesn't touch the running training process.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def hash_file(path, block_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def check_split_duplicates(data_dir):
    splits = ["train", "val", "test"]
    hashes = {s: {} for s in splits}
    for s in splits:
        split_dir = Path(data_dir) / s
        if not split_dir.exists():
            print(f"  [skip] {split_dir} not found")
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            for img in cls_dir.rglob("*"):
                if img.is_file():
                    hashes[s][hash_file(img)] = str(img)

    print("=== Split duplicate check ===")
    for s in splits:
        print(f"  {s}: {len(hashes[s])} unique files")

    dupes = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        common = set(hashes[a]) & set(hashes[b])
        for h in common:
            dupes.append((a, hashes[a][h], b, hashes[b][h]))

    if dupes:
        print(f"  !! {len(dupes)} EXACT duplicate image(s) found across splits:")
        for d in dupes[:10]:
            print(f"     {d[0]}:{d[1]}  ==  {d[2]}:{d[3]}")
        if len(dupes) > 10:
            print(f"     ... and {len(dupes) - 10} more")
    else:
        print("  OK — no exact duplicate files across train/val/test.")
    return dupes


def _get(entry, *keys):
    for k in keys:
        if k in entry and entry[k] is not None:
            return entry[k]
    return None


def check_train_val_gap(history_path, warn_gap=0.05, warn_jump=0.05):
    print(f"\n=== Train/val trajectory ({history_path}) ===")
    with open(history_path) as f:
        raw = json.load(f)
    epochs = raw.get("epochs", raw) if isinstance(raw, dict) else raw
    if not isinstance(epochs, list):
        print("  Unrecognized training_history.json structure — inspect it manually:")
        print(f"  {json.dumps(raw, indent=2)[:500]}")
        return []

    flagged = []
    prev_val_acc = None
    for e in epochs:
        ep = _get(e, "epoch")
        ta = _get(e, "train_acc", "train_accuracy")
        va = _get(e, "val_acc", "val_accuracy")
        tl = _get(e, "train_loss")
        vl = _get(e, "val_loss")
        notes = []

        if ta is not None and va is not None and (ta - va) > warn_gap:
            notes.append("train notably ahead of val (overfitting signature)")
        if prev_val_acc is not None and va is not None and (va - prev_val_acc) > warn_jump:
            notes.append(f"val_acc jumped +{va - prev_val_acc:.3f} in one epoch — check why")
        if va is not None:
            prev_val_acc = va

        marker = f"  <-- {'; '.join(notes)}" if notes else ""
        if notes:
            flagged.append(ep)

        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"

        print(f"  epoch {ep!s:>3}: train_loss={fmt(tl)} val_loss={fmt(vl)} "
              f"train_acc={fmt(ta)} val_acc={fmt(va)}{marker}")

    if not flagged:
        print(f"  OK — no epoch shows a >{warn_gap:.0%} train/val gap or a "
              f">{warn_jump:.0%} single-epoch val_acc jump.")
    else:
        print(f"  Flagged epochs: {flagged}")
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="e.g. resnet18, efficientnet_b3")
    ap.add_argument("--data-dir", default="data/frames",
                     help="root containing train/val/test class folders")
    ap.add_argument("--history", default=None,
                     help="path to training_history.json (auto-detected if omitted)")
    args = ap.parse_args()

    candidates = [
        args.history,
        f"logs/{args.arch}/training_history.json",
        f"checkpoints/{args.arch}/training_history.json",
        "logs/training_history.json",
    ]
    history_path = next((c for c in candidates if c and Path(c).exists()), None)

    dupes = check_split_duplicates(args.data_dir)
    flagged = []
    if history_path:
        flagged = check_train_val_gap(history_path)
    else:
        print(f"\n[!] Could not find training_history.json for '{args.arch}' — "
              f"tried: {[c for c in candidates if c]}")

    print("\n=== Verdict ===")
    if not dupes and not flagged:
        print("No leakage or overfitting signature found in these checks.")
        print("Still confirm with:")
        print("  (1) the held-out TEST set score, not this val score")
        print("  (2) an out-of-distribution check: run best.pth on a handful of")
        print("      images that are NOT part of this Kaggle dataset at all")
    else:
        print("Investigate the flags above before trusting the reported accuracy.")
        sys.exit(1)


if __name__ == "__main__":
    main()
