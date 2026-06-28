"""
tests/fixtures/group_photo_generator.py

Generates synthetic group photo test images containing multiple face-like
ellipses with skin-tone fills. Used to verify the multi-face pipeline
without requiring real photos.

Usage:
    python tests/fixtures/group_photo_generator.py
    # Writes: tests/fixtures/group_2face.jpg
    #         tests/fixtures/group_6face.jpg
    #         tests/fixtures/group_noface.jpg
"""
import math
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def _draw_simple_face(
    canvas: np.ndarray,
    cx: int, cy: int,
    radius: int,
    skin_color: Tuple[int, int, int] = (210, 170, 130),
) -> None:
    """Draw a very simple face oval (enough to trigger MediaPipe) on canvas."""
    # Head
    cv2.ellipse(canvas, (cx, cy), (radius, int(radius * 1.25)),
                0, 0, 360, skin_color, -1)
    # Eyes
    eye_y  = cy - radius // 4
    eye_dx = radius // 3
    for ex in [cx - eye_dx, cx + eye_dx]:
        cv2.circle(canvas, (ex, eye_y), max(2, radius // 8), (60, 40, 30), -1)
    # Nose
    cv2.circle(canvas, (cx, cy + radius // 8), max(2, radius // 12),
               (180, 130, 100), -1)
    # Mouth
    cv2.ellipse(canvas, (cx, cy + radius // 3), (radius // 4, radius // 8),
                0, 0, 180, (140, 80, 70), 2)


def generate_group_photo(
    n_faces: int,
    width: int = 800,
    height: int = 600,
    seed: int = 42,
) -> Image.Image:
    """
    Generate a synthetic image with n_faces simple face drawings.
    Faces are arranged in a rough grid.
    """
    rng = random.Random(seed)
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 230  # light background

    # Add some noise to look less uniform
    noise = np.random.RandomState(seed).randint(0, 15,
                (height, width, 3), dtype=np.uint8)
    canvas = np.clip(canvas.astype(int) + noise - 7, 0, 255).astype(np.uint8)

    if n_faces == 0:
        return Image.fromarray(canvas)

    # Grid layout
    cols  = math.ceil(math.sqrt(n_faces))
    rows  = math.ceil(n_faces / cols)
    cell_w = width  // cols
    cell_h = height // rows
    radius = min(cell_w, cell_h) // 3

    for i in range(n_faces):
        row = i // cols
        col = i  % cols
        cx  = col * cell_w + cell_w // 2 + rng.randint(-5, 5)
        cy  = row * cell_h + cell_h // 2 + rng.randint(-5, 5)
        skin = (
            rng.randint(170, 230),
            rng.randint(130, 190),
            rng.randint(90,  150),
        )
        _draw_simple_face(canvas, cx, cy, radius, skin)

    return Image.fromarray(canvas)


if __name__ == "__main__":
    out_dir = Path("tests/fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("group_2face.jpg",  2),
        ("group_6face.jpg",  6),
        ("group_12face.jpg", 12),
        ("group_noface.jpg", 0),
    ]
    for fname, n in cases:
        img = generate_group_photo(n_faces=n)
        img.save(out_dir / fname, quality=92)
        print(f"[gen] {fname}  ({n} faces)")
