"""
run_batch.py — headless batch video inference entry point.

Reads all settings from config.yaml, then:
  1. Reads the clip selection manifest (dataset/selected_clips.json)
  2. Pre-computes tilt / PSF matrices (cached after first run)
  3. Applies turbulence to every selected video clip, frame by frame
  4. Saves output videos to output_path
  5. Writes a timing summary CSV

Usage
-----
    python run_batch.py [--config config.yaml]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from turbStats import tilt_mat, corr_mat


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Matrix pre-computation (idempotent)
# ─────────────────────────────────────────────────────────────────────────────

def _corr_matrix_exists(data_path: str, corr: float) -> bool:
    return os.path.isfile(os.path.join(data_path, f"R-corr_{corr}.npy"))


def _tilt_matrix_exists(data_path: str, img_size: int, Dr0: float) -> bool:
    return os.path.isfile(
        os.path.join(data_path, f"S_half-size_{img_size}-D_r0_{Dr0:.4f}.npy")
    )


def precompute_matrices(img_size: int, D_over_r0: float, L: float,
                        corr: float, data_path: str) -> None:
    """Generate tilt and PSF matrices if not already cached."""
    D  = 0.1
    r0 = D / D_over_r0

    if not _corr_matrix_exists(data_path, corr):
        print(f"[setup] Generating PSF correlation matrix R-corr_{corr}.npy")
        print("        (can take ~10 min — only once per corr value)")
        corr_mat(corr, data_path)
        print("[setup] PSF matrix done.")
    else:
        print(f"[setup] PSF matrix for corr={corr} already cached.")

    if not _tilt_matrix_exists(data_path, img_size, D_over_r0):
        print(f"[setup] Generating tilt matrix for size={img_size}, D/r0={D_over_r0:.4f}")
        print("        (can take a few minutes — only once per parameter set)")
        tilt_mat(img_size, D, r0, L, data_path)
        print("[setup] Tilt matrix done.")
    else:
        print(f"[setup] Tilt matrix for size={img_size}, D/r0={D_over_r0:.4f} already cached.")


# ─────────────────────────────────────────────────────────────────────────────
# Video sequence → video file conversion
# ─────────────────────────────────────────────────────────────────────────────

def frames_to_video(seq_dir: Path, video_path: Path, fps: float = 30.0) -> bool:
    """
    Assemble JPEG frames from seq_dir into an mp4 using OpenCV.
    Returns True on success, False if no frames found.
    """
    import cv2
    import numpy as np

    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in exts)
    if not frames:
        return False

    # Read first frame to get dimensions
    first = cv2.imread(str(frames[0]))
    if first is None:
        return False
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        return False

    for p in frames:
        frame = cv2.imread(str(p))
        if frame is not None:
            writer.write(frame)
    writer.release()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    data_path    = Path(cfg["data_path"])
    dataset_path = Path(cfg["dataset_path"])
    output_path  = Path(cfg["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_path.parent / "selected_clips.json"

    # ── Device ───────────────────────────────────────────────────────────────
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available — falling back to CPU.")
        device_str = "cpu"
    print(f"[info] Device: {device_str}")

    # ── Turbulence params ────────────────────────────────────────────────────
    D_over_r0 = float(cfg["D_over_r0"])
    scale     = float(cfg.get("scale", 1.0))
    corr      = float(cfg.get("corr", -0.1))
    L         = float(cfg.get("L", 3000.0))
    img_size  = int(cfg.get("img_size", 256))
    D         = 0.1
    r0        = D / D_over_r0

    print(f"[info] D/r0={D_over_r0}  scale={scale}  corr={corr}  "
          f"L={L}  img_size={img_size}")

    # ── Pre-compute matrices ─────────────────────────────────────────────────
    precompute_matrices(img_size, D_over_r0, L, corr, str(data_path))

    # ── Load clip manifest ────────────────────────────────────────────────────
    if not manifest_path.exists():
        print(f"[error] Selection manifest not found: {manifest_path}")
        print("        Run: python select_diverse_clips.py --dataset <path>")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    selected_clips = manifest["selected_clips"]
    print(f"\n[info] Processing {len(selected_clips)} selected clips → {output_path}")

    # ── Scratch folder for assembled input videos ─────────────────────────────
    tmp_dir = dataset_path.parent / "_tmp_videos"
    tmp_dir.mkdir(exist_ok=True)

    # ── Process each clip ─────────────────────────────────────────────────────
    results = []
    t_start = time.time()

    for i, clip_name in enumerate(selected_clips):
        seq_dir = dataset_path / "sequences" / clip_name

        if not seq_dir.exists():
            print(f"  [skip] Sequence folder not found: {seq_dir}")
            continue

        print(f"\n[{i+1}/{len(selected_clips)}] {clip_name}")

        # Assemble frames → temp input video
        tmp_input = tmp_dir / f"{clip_name}.mp4"
        if not tmp_input.exists():
            print(f"  Assembling frames from {seq_dir} …")
            ok = frames_to_video(seq_dir, tmp_input, fps=30.0)
            if not ok:
                print(f"  [skip] No frames found in {seq_dir}")
                continue
        else:
            print(f"  Using existing temp video: {tmp_input}")

        out_video = output_path / f"{clip_name}_turbulence.mp4"

        # Call apply_turbulence_to_video.py as a subprocess so the
        # simulator runs in a clean state for each clip (avoids GPU
        # memory fragmentation on long runs).
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "apply_turbulence_to_video.py"),
            "--input",    str(tmp_input),
            "--output",   str(out_video),
            "--D",        str(D),
            "--r0",       str(r0),
            "--L",        str(L),
            "--img_size", str(img_size),
            "--corr",     str(corr),
            "--scale",    str(scale),
        ]

        t0 = time.time()
        print(f"  Running: {' '.join(cmd)}", flush=True)
        ret = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
        elapsed = time.time() - t0

        status = "ok" if ret.returncode == 0 else f"error({ret.returncode})"
        results.append({
            "clip":    clip_name,
            "output":  out_video.name,
            "time_s":  f"{elapsed:.1f}",
            "status":  status,
        })
        print(f"  {status} — {elapsed:.1f}s → {out_video.name}")

    # ── Clean up temp videos ─────────────────────────────────────────────────
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    total = time.time() - t_start
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"\n[done] {ok_count}/{len(results)} clips processed in {total:.1f}s")

    # ── Summary CSV ──────────────────────────────────────────────────────────
    csv_path = output_path / "results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "output", "time_s", "status"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[info] Summary written to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless batch video turbulence inference")
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML (default: config.yaml)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    main(cfg)
