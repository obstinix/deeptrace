#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, log_file):
    print(f"Running command: {' '.join(cmd)}")
    print(f"Redirecting output to: {log_file}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # We pass the current environment and force PYTHONUNBUFFERED=1
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "src"
    
    with open(log_file, "w", encoding="utf-8") as f:
        # Run process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1
        )
        # Stream output to both file and console
        for line in process.stdout:
            f.write(line)
            f.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            
        process.wait()
        return process.returncode

def main():
    configs = [
        ("resnet18", "training/configs/resnet18.yaml"),
        ("efficientnet_b3", "training/configs/efficientnet_b3.yaml"),
        ("vit_base", "training/configs/vit_base.yaml"),
        ("efficientnet_b0", "training/configs/efficientnet_b0.yaml"),
        ("vit_b16", "training/configs/vit_b16.yaml"),
    ]
    
    for arch, config_path in configs:
        print(f"\n=========================================")
        print(f"STARTING TRAINING FOR {arch}")
        print(f"=========================================\n")
        
        # We always pass --resume in case it was interrupted
        cmd = [
            sys.executable,
            "training/train.py",
            "--config", config_path,
            "--resume"
        ]
        log_file = Path("logs") / arch / "train.log"
        
        ret = run_command(cmd, log_file)
        if ret != 0:
            print(f"\n[ERROR] Training for {arch} failed with exit code {ret}!")
        else:
            print(f"\n[SUCCESS] Training for {arch} completed successfully.")

if __name__ == "__main__":
    main()
