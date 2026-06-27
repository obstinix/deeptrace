#!/usr/bin/env python3
"""
scripts/generate_synthetic_videos.py
Generates a small mock dataset of real and fake videos to test the training pipeline.
Real videos contain a green circle in the center.
Fake videos contain a red square in the center.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def generate_video(path: Path, is_real: bool, num_frames: int = 30, size: tuple[int, int] = (224, 224)):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, size)

    for i in range(num_frames):
        # Base background (noise or static)
        frame = np.random.randint(100, 150, (size[1], size[0], 3), dtype=np.uint8)

        if is_real:
            # Draw a green circle
            cv2.circle(frame, (size[0] // 2, size[1] // 2), 40, (0, 255, 0), -1)
        else:
            # Draw a red square
            cv2.rectangle(frame, (size[0] // 2 - 40, size[1] // 2 - 40),
                          (size[0] // 2 + 40, size[1] // 2 + 40), (0, 0, 255), -1)

        out.write(frame)

    out.release()


def main():
    raw_dir = Path("data/raw")
    
    # Define folders
    real_dirs = [raw_dir / "Celeb-real", raw_dir / "YouTube-real"]
    fake_dirs = [raw_dir / "Celeb-synthesis"]
    
    # Generate 15 videos per folder (total 45 videos)
    n_videos = 15
    
    print("Generating synthetic videos...")
    
    # Real
    for r_dir in real_dirs:
        print(f"Generating real videos in {r_dir.name}...")
        for i in tqdm(range(n_videos)):
            v_path = r_dir / f"real_video_{i:03d}.mp4"
            generate_video(v_path, is_real=True)
            
    # Fake
    for f_dir in fake_dirs:
        print(f"Generating fake videos in {f_dir.name}...")
        for i in tqdm(range(n_videos)):
            v_path = f_dir / f"fake_video_{i:03d}.mp4"
            generate_video(v_path, is_real=False)

    print("\nSynthetic video dataset successfully generated!")


if __name__ == "__main__":
    main()
