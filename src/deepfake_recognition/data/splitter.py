"""Stratified train/val/test split utility."""
from __future__ import annotations
import random
from pathlib import Path


def stratified_split(
    samples: list[tuple[Path, int]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, list[tuple[Path, int]]]:
    """
    Split (path, label) samples into train/val/test with per-class stratification.
    Returns dict with keys 'train', 'val', 'test'.
    """
    by_class: dict[int, list] = {}
    for item in samples:
        label = item[1]
        by_class.setdefault(label, []).append(item)

    rng = random.Random(seed)
    result: dict[str, list] = {"train": [], "val": [], "test": []}

    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        result["test"].extend(items[:n_test])
        result["val"].extend(items[n_test: n_test + n_val])
        result["train"].extend(items[n_test + n_val:])

    for split in result:
        rng.shuffle(result[split])

    return result
