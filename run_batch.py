"""
run_batch.py — headless batch inference entry point.

Reads all settings from config.yaml, then processes BOTH:

  • Images  (visdrone_det_val)  → turbulence applied per image via the
    Simulator directly; outputs saved to  output_path/images/

  • Videos  (visdrone_vid_val)  → diversity-selected clips processed
    frame-by-frame via apply_turbulence_to_video.py; outputs saved to
    output_path/videos/

Usage
-----
    python run_batch.py [--config config.yaml]
                        [--images-only | --videos-only]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
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
# Image batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_images(cfg: dict, output_path: Path) -> list:
    """
    Apply turbulence to every image in the DET-val dataset.
    Images are read from  dataset_det_path  (or the images/ sub-folder).
    Results are written to  output_path/images/
    """
    from simulator import Simulator

    det_path = Path(cfg.get("dataset_det_path", "./dataset/visdrone_det_val"))
    img_out  = output_path / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # Locate image files — try  det_path/images/  first, then root
    search_dir = det_path / "images" if (det_path / "images").exists() else det_path
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(p for p in search_dir.iterdir() if p.suffix.lower() in exts)

    if not image_files:
        print(f"[images] No images found in {search_dir} — skipping image batch.")
        return []

    print(f"[images] Found {len(image_files)} images in {search_dir}")

    # ── Build simulator ──────────────────────────────────────────────────────
    data_path  = Path(cfg["data_path"])
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    D_over_r0 = float(cfg["D_over_r0"])
    scale     = float(cfg.get("scale", 1.0))
    corr      = float(cfg.get("corr", -0.1))
    img_size  = int(cfg.get("img_size", 256))

    simulator = Simulator(
        Dr0=D_over_r0,
        img_size=img_size,
        corr=corr,
        data_path=str(data_path),
        device=device_str,
        scale=scale,
    ).to(device, dtype=torch.float32)
    simulator.eval()

    results = []
    t_start = time.time()

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            t0 = time.time()
            try:
                # Read image
                frame_bgr = cv2.imread(str(img_path))
                if frame_bgr is None:
                    raise IOError(f"Cannot read {img_path}")

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                src_h, src_w = frame_rgb.shape[:2]

                # Resize to simulator input size
                frame_sq = cv2.resize(frame_rgb, (img_size, img_size),
                                      interpolation=cv2.INTER_LINEAR)

                # To tensor
                img_t = (torch.from_numpy(frame_sq)
                         .to(device, dtype=torch.float32)
                         .permute(2, 0, 1) / 255.0)

                # Apply turbulence
                out_t = simulator(img_t)

                # Back to numpy
                out_np = out_t.cpu().numpy()
                if out_np.ndim == 3:
                    out_np = out_np.transpose(1, 2, 0)
                out_np = np.clip(out_np, 0.0, 1.0)
                out_uint8 = (out_np * 255).astype(np.uint8)

                # Resize back to original resolution
                if (img_size, img_size) != (src_h, src_w):
                    out_uint8 = cv2.resize(out_uint8, (src_w, src_h),
                                           interpolation=cv2.INTER_LINEAR)

                out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
                out_name = img_path.stem + "_turbulence" + img_path.suffix
                out_file = img_out / out_name
                cv2.imwrite(str(out_file), out_bgr)

                elapsed = time.time() - t0
                status  = "ok"

            except Exception as exc:
                elapsed = time.time() - t0
                status  = f"error({exc})"
                out_name = img_path.name

            results.append({
                "file":   img_path.name,
                "output": out_name,
                "time_s": f"{elapsed:.2f}",
                "status": status,
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
                ok = sum(1 for r in results if r["status"] == "ok")
                print(f"  [{i+1}/{len(image_files)}] {ok} ok  "
                      f"(last: {results[-1]['status']})", flush=True)

    total = time.time() - t_start
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"[images] Done: {ok_count}/{len(results)} images in {total:.1f}s → {img_out}")

    # ── Summary CSV ──────────────────────────────────────────────────────────
    csv_path = img_out / "results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "output", "time_s", "status"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[images] Summary → {csv_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Video batch processing
# ─────────────────────────────────────────────────────────────────────────────

def frames_to_video(seq_dir: Path, video_path: Path, fps: float = 30.0) -> bool:
    """Assemble JPEG frames into an mp4. Returns True on success."""
    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in exts)
    if not frames:
        return False

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


def process_videos(cfg: dict, output_path: Path) -> list:
    """
    Apply turbulence to every selected video clip in the VID-val dataset.
    Clips are read from  dataset_path/sequences/  (as frame folders).
    Results are written to  output_path/videos/
    """
    import shutil

    dataset_path = Path(cfg["dataset_path"])
    vid_out      = output_path / "videos"
    vid_out.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_path.parent / "selected_clips.json"
    if not manifest_path.exists():
        print(f"[videos] Manifest not found: {manifest_path}")
        print("         Run: python select_diverse_clips.py --dataset <path>")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    selected_clips = manifest["selected_clips"]
    print(f"[videos] Processing {len(selected_clips)} selected clips → {vid_out}")

    data_path = Path(cfg["data_path"])
    scale     = float(cfg.get("scale", 1.0))
    D         = 0.1
    min_sec   = float(cfg.get("param_change_min_seconds", 1))
    max_sec   = float(cfg.get("param_change_max_seconds", 30))

    tmp_dir = dataset_path.parent / "_tmp_videos"
    tmp_dir.mkdir(exist_ok=True)

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

        out_video = vid_out / f"{clip_name}_turbulence.mp4"

        # apply_turbulence_to_video.py handles img_size=1024, varying params
        # internally — no need to pass D_over_r0 / corr / L / img_size here.
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "apply_turbulence_to_video.py"),
            "--input",   str(tmp_input),
            "--output",  str(out_video),
            "--D",       str(D),
            "--scale",   str(scale),
            "--param-change-min-seconds", str(min_sec),
            "--param-change-max-seconds", str(max_sec),
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

    # Clean up temp videos
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    total = time.time() - t_start
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"\n[videos] Done: {ok_count}/{len(results)} clips in {total:.1f}s → {vid_out}")

    # ── Summary CSV ──────────────────────────────────────────────────────────
    csv_path = vid_out / "results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "output", "time_s", "status"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[videos] Summary → {csv_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: dict, images_only: bool = False, videos_only: bool = False) -> None:
    data_path   = Path(cfg["data_path"])
    output_path = Path(cfg["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available — falling back to CPU.")
        device_str = "cpu"
        cfg["device"] = device_str
    print(f"[info] Device: {device_str}")

    # ── Turbulence params ────────────────────────────────────────────────────
    D_over_r0 = float(cfg["D_over_r0"])
    scale     = float(cfg.get("scale", 1.0))
    corr      = float(cfg.get("corr", -0.1))
    L         = float(cfg.get("L", 3000.0))
    img_size  = int(cfg.get("img_size", 256))

    print(f"[info] D/r0={D_over_r0}  scale={scale}  corr={corr}  "
          f"L={L}  img_size={img_size}")

    # ── Pre-compute matrices (shared by both image and video pipelines) ───────
    precompute_matrices(img_size, D_over_r0, L, corr, str(data_path))

    # ── Run pipelines ─────────────────────────────────────────────────────────
    if not videos_only:
        print("\n" + "=" * 60)
        print("  IMAGE PIPELINE")
        print("=" * 60)
        process_images(cfg, output_path)

    if not images_only:
        print("\n" + "=" * 60)
        print("  VIDEO PIPELINE")
        print("=" * 60)
        process_videos(cfg, output_path)

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: {output_path}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless batch turbulence inference (images + videos)")
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML (default: config.yaml)"
    )
    p.add_argument("--images-only", action="store_true",
                   help="Only process the image dataset")
    p.add_argument("--videos-only", action="store_true",
                   help="Only process the video dataset")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    main(cfg, images_only=args.images_only, videos_only=args.videos_only)
