#!/usr/bin/env python3
"""
scripts/prepare_dataset.py
Prepare Celeb-DF v2 dataset by extracting frames at video-level.
"""
import argparse
import random
from pathlib import Path
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from Celeb-DF v2 videos")
    parser.add_argument("--input", default="data/raw", type=Path, help="Path to raw dataset")
    parser.add_argument("--output", default="data/frames", type=Path, help="Path to output frames")
    parser.add_argument("--fps", default=1.0, type=float, help="Frames per second to extract")
    parser.add_argument("--max-frames", default=30, type=int, help="Max frames to extract per video")
    parser.add_argument("--real-dirs", default="Celeb-real,YouTube-real", help="Comma-separated real video dirs")
    parser.add_argument("--fake-dirs", default="Celeb-synthesis", help="Comma-separated fake video dirs")
    return parser.parse_args()


def get_videos(root_dir: Path, subdirs: list[str]) -> list[Path]:
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = []
    for sd in subdirs:
        sd_path = root_dir / sd
        if not sd_path.exists():
            print(f"[warning] Directory not found: {sd_path}")
            continue
        for p in sd_path.rglob("*"):
            if p.suffix.lower() in video_exts:
                videos.append(p)
    return videos


def extract_frames_from_video(video_path: Path, dest_dir: Path, target_fps: float, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warning] Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None or str(fps) == "nan":
        fps = 30.0  # fallback

    frame_step = max(1, int(round(fps / target_fps)))
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    video_name = video_path.stem
    parent_name = video_path.parent.name
    
    frame_idx = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            frame_filename = f"{parent_name}_{video_name}_f{extracted_count:02d}.jpg"
            out_path = dest_dir / frame_filename
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            extracted_count += 1
            if extracted_count >= max_frames:
                break
        frame_idx += 1
        
    cap.release()
    return extracted_count


def main():
    args = parse_args()
    
    real_subdirs = [d.strip() for d in args.real_dirs.split(",") if d.strip()]
    fake_subdirs = [d.strip() for d in args.fake_dirs.split(",") if d.strip()]
    
    real_videos = get_videos(args.input, real_subdirs)
    fake_videos = get_videos(args.input, fake_subdirs)
    
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos.")
    
    if not real_videos and not fake_videos:
        print("[error] No videos found. Check --input directory and subdirectories.")
        return
        
    # Split datasets deterministically on a video-level
    random.seed(42)
    
    # Real
    real_videos.sort()
    random.shuffle(real_videos)
    n_real = len(real_videos)
    n_real_train = int(0.8 * n_real)
    n_real_val = int(0.1 * n_real)
    
    real_splits = {
        "train": real_videos[:n_real_train],
        "val": real_videos[n_real_train:n_real_train + n_real_val],
        "test": real_videos[n_real_train + n_real_val:]
    }
    
    # Fake
    fake_videos.sort()
    random.shuffle(fake_videos)
    n_fake = len(fake_videos)
    n_fake_train = int(0.8 * n_fake)
    n_fake_val = int(0.1 * n_fake)
    
    fake_splits = {
        "train": fake_videos[:n_fake_train],
        "val": fake_videos[n_fake_train:n_fake_train + n_fake_val],
        "test": fake_videos[n_fake_train + n_fake_val:]
    }
    
    counts = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}, "test": {"real": 0, "fake": 0}}
    
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} split...")
        
        # Process real videos in split
        real_dest = args.output / split / "real"
        for v_path in tqdm(real_splits[split], desc=f"Real ({split})"):
            c = extract_frames_from_video(v_path, real_dest, args.fps, args.max_frames)
            counts[split]["real"] += c
            
        # Process fake videos in split
        fake_dest = args.output / split / "fake"
        for v_path in tqdm(fake_splits[split], desc=f"Fake ({split})"):
            c = extract_frames_from_video(v_path, fake_dest, args.fps, args.max_frames)
            counts[split]["fake"] += c
            
    print("\nFrame Extraction Complete!")
    for split in ["train", "val", "test"]:
        r_c = counts[split]["real"]
        f_c = counts[split]["fake"]
        tot = r_c + f_c
        print(f"  {split.upper()}:")
        print(f"    Real frames: {r_c:,}")
        print(f"    Fake frames: {f_c:,}")
        print(f"    Total:       {tot:,}")


if __name__ == "__main__":
    main()
