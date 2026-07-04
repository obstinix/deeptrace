#!/usr/bin/env python3
"""
Automatic deepfake dataset downloader.

Downloads one of these datasets (in order of preference):
  1. 140k Real and Fake Faces (Kaggle, ~2GB, no approval needed)
  2. Real and Fake Face Detection (Kaggle, ~500MB, no approval needed)
  3. DFDC Sample (Kaggle, requires competition join — free)
  4. Synthetic fallback (generates tiny dataset for smoke-testing, no download)

Usage:
  # Download best available dataset automatically
  python scripts/download_dataset.py auto --output data/frames

  # Download specific dataset
  python scripts/download_dataset.py kaggle-140k --output data/frames
  python scripts/download_dataset.py kaggle-real-fake --output data/frames
  python scripts/download_dataset.py dfdc --output data/frames

  # Generate tiny synthetic dataset (for smoke testing, no internet needed)
  python scripts/download_dataset.py synthetic --output data/frames --n-images 200

  # Setup Kaggle credentials interactively
  python scripts/download_dataset.py setup-kaggle

Kaggle credentials:
  Option A (recommended): Set env vars
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
  Option B: Place kaggle.json at ~/.kaggle/kaggle.json
    {"username":"your_username","key":"your_api_key"}
  Get your API key: https://www.kaggle.com/settings → API → Create New Token
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


# ─── Kaggle helpers ───────────────────────────────────────────────────────────

def _get_kaggle_creds() -> tuple[str, str] | None:
    """Return (username, key) from env or ~/.kaggle/kaggle.json, or None."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return "token", token

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json) as f:
            creds = json.load(f)
        if "token" in creds:
             return "token", creds["token"]
        if creds.get("key", "").startswith("KGAT_"):
             return "token", creds["key"]
        return creds.get("username"), creds.get("key")

    return None


def _kaggle_download(dataset: str, dest: Path, is_competition: bool = False) -> Path:
    """
    Download a Kaggle dataset or competition data.
    dataset: "owner/dataset-name" or "competition-name"
    dest: directory to save zip
    Returns path to downloaded zip.
    """
    creds = _get_kaggle_creds()
    if creds is None:
        raise RuntimeError(
            "Kaggle credentials not found.\n"
            "Run: python scripts/download_dataset.py setup-kaggle\n"
            "Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
    username, key = creds

    if is_competition:
        url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{dataset}"
    else:
        url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "dataset.zip"

    print(f"Downloading from Kaggle: {dataset}")
    print(f"Destination: {zip_path}")

    if username == "token" or key.startswith("KGAT_"):
        headers = {"Authorization": f"Bearer {key}"}
        response = requests.get(url, headers=headers, stream=True, timeout=120)
    else:
        response = requests.get(url, auth=(username, key), stream=True, timeout=120)

    if response.status_code == 403:
        raise RuntimeError(
            f"Access denied (403). For competition datasets, you must first\n"
            f"join the competition at https://www.kaggle.com/c/{dataset}\n"
            f"(free, just click 'Join Competition')"
        )
    if response.status_code == 401:
        raise RuntimeError("Invalid Kaggle credentials. Check your username/key.")
    if response.status_code != 200:
        raise RuntimeError(f"Download failed: HTTP {response.status_code}")

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return zip_path


# ─── Dataset processors ───────────────────────────────────────────────────────

def download_140k(output: Path):
    """
    140k Real and Fake Faces (StyleGAN2 generated)
    Kaggle: xhlulu/140k-real-and-fake-faces
    Size: ~2GB
    Structure after download:
      real_vs_fake/real-vs-fake/
        train/real/, train/fake/
        valid/real/, valid/fake/
        test/real/,  test/fake/
    """
    raw = output.parent / "raw_140k"
    zip_path = _kaggle_download("xhlulu/140k-real-and-fake-faces", raw)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw)

    copied = {"real": 0, "fake": 0}
    for split in ["train", "valid", "test"]:
        dst_split = "val" if split == "valid" else split
        for cls in ["real", "fake"]:
            src = raw / "real_vs_fake" / "real-vs-fake" / split / cls
            if not src.exists():
                # Try alternate path structure
                src = raw / split / cls
            if not src.exists():
                continue
            dst = output / dst_split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for img in tqdm(list(src.glob("*.jpg")) + list(src.glob("*.png")),
                            desc=f"{split}/{cls}"):
                shutil.copy2(img, dst / img.name)
                copied[cls] += 1

    shutil.rmtree(raw, ignore_errors=True)
    zip_path.unlink(missing_ok=True)

    deduplicate_dataset(output)
    _print_stats(output, copied)


def download_real_fake(output: Path):
    """
    Real and Fake Face Detection (Kaggle)
    Dataset: ciplab/real-and-fake-face-detection
    Size: ~500MB
    Contains: real faces + 3 categories of fake (easy/mid/hard)
    """
    raw = output.parent / "raw_real_fake"
    zip_path = _kaggle_download("ciplab/real-and-fake-face-detection", raw)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw)

    real_out = output / "real"
    fake_out = output / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    copied = {"real": 0, "fake": 0}

    # Real faces
    for src_dir in raw.rglob("*"):
        if not src_dir.is_dir():
            continue
        name = src_dir.name.lower()
        if "real" in name:
            dst = real_out
            cls = "real"
        elif "fake" in name or "easy" in name or "mid" in name or "hard" in name:
            dst = fake_out
            cls = "fake"
        else:
            continue
        for img in src_dir.glob("*.jpg"):
            shutil.copy2(img, dst / f"{src_dir.name}_{img.name}")
            copied[cls] += 1

    zip_path.unlink()
    _print_stats(output, copied)


def download_dfdc(output: Path):
    """
    DFDC Sample dataset (Kaggle competition — requires free join).
    Competition: deepfake-detection-challenge
    Downloads the sample submission + metadata.

    NOTE: User must join competition at:
    https://www.kaggle.com/c/deepfake-detection-challenge
    (free, instant, just click Join Competition)
    """
    raw = output.parent / "raw_dfdc"
    zip_path = _kaggle_download("deepfake-detection-challenge", raw, is_competition=True)

    print("Extracting DFDC...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw)

    # Find metadata.json
    meta_files = list(raw.rglob("metadata.json"))
    if not meta_files:
        raise RuntimeError("No metadata.json found in DFDC download.")

    real_out = output / "real"
    fake_out = output / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    copied = {"real": 0, "fake": 0}

    for meta_file in meta_files:
        with open(meta_file) as f:
            metadata = json.load(f)
        videos_dir = meta_file.parent

        for fname, info in metadata.items():
            vid_path = videos_dir / fname
            if not vid_path.exists():
                continue
            label = info.get("label", "").upper()
            # Extract frames from video
            try:
                import cv2
                cap = cv2.VideoCapture(str(vid_path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                dst = real_out if label == "REAL" else fake_out
                cls = "real" if label == "REAL" else "fake"
                for i in range(min(8, total)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / 8))
                    ret, frame = cap.read()
                    if ret:
                        out_path = dst / f"{vid_path.stem}_f{i:02d}.jpg"
                        import cv2 as _cv2
                        _cv2.imwrite(str(out_path), frame, [_cv2.IMWRITE_JPEG_QUALITY, 90])
                        copied[cls] += 1
                cap.release()
            except Exception as e:
                print(f"  Skipping {fname}: {e}")

    zip_path.unlink(missing_ok=True)
    _print_stats(output, copied)


def generate_synthetic(output: Path, n_images: int = 200):
    """
    Generate a tiny synthetic dataset for smoke-testing the pipeline.
    No internet connection or Kaggle account needed.
    Creates n_images/2 real-like and n_images/2 fake-like face images.

    NOT for training real models — only for pipeline testing.
    Images are random noise patches (enough to verify data loading works).
    """
    import random
    import numpy as np

    real_out = output / "real"
    fake_out = output / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    n_each = n_images // 2
    print(f"Generating {n_each} synthetic real + {n_each} fake images...")
    print("(These are noise images for pipeline testing ONLY — not for real training)")

    rng = random.Random(42)

    for i in tqdm(range(n_each), desc="Synthetic real"):
        # Warm-toned noise (simulates skin tones)
        arr = np.random.randint(100, 220, (224, 224, 3), dtype=np.uint8)
        arr[:, :, 2] = arr[:, :, 2] * 0.7  # reduce blue
        img = Image.fromarray(arr)
        img.save(real_out / f"synthetic_real_{i:04d}.jpg", quality=85)

    for i in tqdm(range(n_each), desc="Synthetic fake"):
        # Cool-toned noise with slightly different distribution
        arr = np.random.randint(80, 200, (224, 224, 3), dtype=np.uint8)
        arr[:, :, 0] = arr[:, :, 0] * 0.8  # reduce red
        img = Image.fromarray(arr)
        img.save(fake_out / f"synthetic_fake_{i:04d}.jpg", quality=85)

    print(f"\nSynthetic dataset created at {output}")
    print("Run: python scripts/download_dataset.py verify --path data/frames")
    print("Then: python training/train.py --config training/configs/resnet18.yaml"
          " --data data/frames --max-samples 100")


# ─── Auto mode ────────────────────────────────────────────────────────────────

def auto_download(output: Path):
    """
    Try datasets in order until one succeeds.
    Order: 140k faces → real-fake detection → DFDC → synthetic fallback
    """
    creds = _get_kaggle_creds()

    if creds is None:
        print("No Kaggle credentials found.")
        print("Falling back to synthetic dataset for smoke-testing.")
        print("For real training, run: python scripts/download_dataset.py setup-kaggle")
        generate_synthetic(output, n_images=400)
        return

    attempts = [
        ("140k Real and Fake Faces", download_140k),
        ("Real and Fake Face Detection", download_real_fake),
    ]

    for name, fn in attempts:
        try:
            print(f"\nAttempting: {name}")
            fn(output)
            return
        except Exception as e:
            print(f"  Failed: {e}")
            print(f"  Trying next option...")

    print("\nAll Kaggle downloads failed. Generating synthetic dataset.")
    generate_synthetic(output, n_images=400)


# ─── Kaggle setup helper ──────────────────────────────────────────────────────

def setup_kaggle():
    """Interactive setup for Kaggle credentials."""
    print("=" * 60)
    print("Kaggle API Setup")
    print("=" * 60)
    print()
    print("1. Go to: https://www.kaggle.com/settings")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New Token'")
    print("4. This downloads kaggle.json")
    print()
    print("Then choose one of:")
    print()
    print("Option A — Environment variables (recommended for agents):")
    print("  export KAGGLE_USERNAME=your_username")
    print("  export KAGGLE_KEY=your_api_key_here")
    print()
    print("Option B — Place kaggle.json at ~/.kaggle/kaggle.json:")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    print(f"  File location: {kaggle_json}")
    print('  Content: {"username":"YOUR_USERNAME","key":"YOUR_KEY"}')
    print()

    username = input("Enter your Kaggle username (or press Enter to skip): ").strip()
    if username:
        key = input("Enter your Kaggle API key: ").strip()
        if key:
            kaggle_json.write_text(json.dumps({"username": username, "key": key}))
            kaggle_json.chmod(0o600)
            print(f"\nCredentials saved to {kaggle_json}")
            print("Test with: python scripts/download_dataset.py auto --output data/frames")
        else:
            print("No key provided. Skipping.")


# ─── Utilities ────────────────────────────────────────────────────────────────

def deduplicate_dataset(output: Path):
    import hashlib
    print("Checking for duplicate images across splits by file hash...")
    seen_hashes = {}
    duplicates_removed = 0
    
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            dir_path = output / split / cls
            if not dir_path.exists():
                continue
            for img_path in list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png")):
                try:
                    with open(img_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in seen_hashes:
                        img_path.unlink()
                        duplicates_removed += 1
                    else:
                        seen_hashes[file_hash] = f"{split}/{cls}/{img_path.name}"
                except Exception as e:
                    print(f"Error checking hash for {img_path}: {e}")
    print(f"Deduplication complete. Removed {duplicates_removed} duplicate images.")


def verify_dataset(path: Path):
    print(f"\nDataset verification at: {path}")
    if (path / "train").exists():
        overall_real = 0
        overall_fake = 0
        for split in ["train", "val", "test"]:
            real_dir = path / split / "real"
            fake_dir = path / split / "fake"
            real_count = len(list(real_dir.glob("*.jpg"))) + len(list(real_dir.glob("*.png"))) if real_dir.exists() else 0
            fake_count = len(list(fake_dir.glob("*.jpg"))) + len(list(fake_dir.glob("*.png"))) if fake_dir.exists() else 0
            split_total = real_count + fake_count
            overall_real += real_count
            overall_fake += fake_count
            print(f"  Split: {split}")
            print(f"    Real images : {real_count:,}")
            print(f"    Fake images : {fake_count:,}")
            print(f"    Total       : {split_total:,}")
            if split_total > 0:
                balance = real_count / split_total
                print(f"    Balance     : {balance:.1%} real / {1-balance:.1%} fake")
        
        total = overall_real + overall_fake
        print(f"  Overall Total: {total:,}")
        if total == 0:
            print("  ERROR: No images found! Check the path.")
            return
        balance = overall_real / total
    else:
        real_dir = path / "real"
        fake_dir = path / "fake"
        real_count = len(list(real_dir.glob("*.jpg"))) + len(list(real_dir.glob("*.png"))) \
                     if real_dir.exists() else 0
        fake_count = len(list(fake_dir.glob("*.jpg"))) + len(list(fake_dir.glob("*.png"))) \
                     if fake_dir.exists() else 0
        total = real_count + fake_count
        print(f"  Real images : {real_count:,}")
        print(f"  Fake images : {fake_count:,}")
        print(f"  Total       : {total:,}")
        if total == 0:
            print("  ERROR: No images found! Check the path.")
            return
        balance = real_count / total
        print(f"  Balance     : {balance:.1%} real / {1-balance:.1%} fake")

    if total < 1000:
        print("  NOTE: Small dataset (<1000 images). Good for smoke testing.")
    elif total < 10000:
        print("  NOTE: Medium dataset. Expect ~80-85% accuracy.")
    else:
        print("  OK: Large dataset. Expect good training results.")

    print()
    print("Ready to train:")
    print(f"  python training/train.py --config training/configs/resnet18.yaml "
          f"--data {path}")


def _print_stats(output: Path, copied: dict):
    print(f"\nDownload complete!")
    print(f"  Real: {copied['real']:,} images")
    print(f"  Fake: {copied['fake']:,} images")
    verify_dataset(output)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deepfake dataset downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_auto = sub.add_parser("auto", help="Download best available dataset automatically")
    p_auto.add_argument("--output", required=True, type=Path)

    p_140k = sub.add_parser("kaggle-140k", help="Download 140k Real/Fake Faces (Kaggle)")
    p_140k.add_argument("--output", required=True, type=Path)

    p_rf = sub.add_parser("kaggle-real-fake", help="Real and Fake Face Detection (Kaggle)")
    p_rf.add_argument("--output", required=True, type=Path)

    p_dfdc = sub.add_parser("dfdc", help="DFDC sample (Kaggle — must join competition first)")
    p_dfdc.add_argument("--output", required=True, type=Path)

    p_syn = sub.add_parser("synthetic", help="Generate tiny synthetic dataset (no internet)")
    p_syn.add_argument("--output", required=True, type=Path)
    p_syn.add_argument("--n-images", type=int, default=200)

    p_setup = sub.add_parser("setup-kaggle", help="Configure Kaggle credentials")

    p_verify = sub.add_parser("verify", help="Verify a dataset directory")
    p_verify.add_argument("--path", required=True, type=Path)

    args = parser.parse_args()

    if args.cmd == "auto":
        auto_download(args.output)
    elif args.cmd == "kaggle-140k":
        download_140k(args.output)
    elif args.cmd == "kaggle-real-fake":
        download_real_fake(args.output)
    elif args.cmd == "dfdc":
        download_dfdc(args.output)
    elif args.cmd == "synthetic":
        generate_synthetic(args.output, args.n_images)
    elif args.cmd == "setup-kaggle":
        setup_kaggle()
    elif args.cmd == "verify":
        verify_dataset(args.path)


if __name__ == "__main__":
    main()
