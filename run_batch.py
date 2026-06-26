"""
run_batch.py — headless batch inference entry point.

Reads all settings from config.yaml, then processes BOTH:

  • Images  (visdrone_det_val)  → turbulence applied per image via the
    Simulator directly; outputs saved to  output_path/images/
    Each image gets a unique random parameter set (D_over_r0, L, corr,
    scale).  A .json label file and master results_summary.csv are written.
    img_size is fixed at 1024 for all images.

  • Videos  (visdrone_vid_val)  → ALL sequences processed frame-by-frame
    via apply_turbulence_to_video.py; outputs saved to  output_path/videos/
    No subset selection — every sequence folder in dataset_path/sequences/
    is processed.

Usage
-----
    python run_batch.py [--config config.yaml]
                        [--images-only | --videos-only]
"""

import argparse
import csv
import json
import os
import random
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

# ── Fixed image resolution (overrides config.yaml img_size for images) ────────
IMAGE_IMG_SIZE = 1024

# ── Random parameter ranges for image pipeline ────────────────────────────────
DR0_RANGE   = (0.1, 3.0)
L_RANGE     = (500.0, 5000.0)
SCALE_RANGE = (0.5, 2.0)
CORR_VALUES = [-0.1, 0.0, 0.5, 0.9]

# ── D (aperture diameter) is fixed at 0.1 m throughout ───────────────────────
D_APERTURE = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Turbulence strength label
# ─────────────────────────────────────────────────────────────────────────────

def strength_label(Dr0: float) -> str:
    if Dr0 < 1.0:
        return "light"
    elif Dr0 <= 2.0:
        return "moderate"
    return "strong"


# ─────────────────────────────────────────────────────────────────────────────
# Matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def _corr_matrix_exists(data_path: str, corr: float) -> bool:
    return os.path.isfile(os.path.join(data_path, f"R-corr_{corr}.npy"))


def _tilt_matrix_path(data_path: str, img_size: int, Dr0: float) -> str:
    return os.path.join(data_path, f"S_half-size_{img_size}-D_r0_{Dr0:.4f}.npy")


def _tilt_matrix_exists(data_path: str, img_size: int, Dr0: float) -> bool:
    return os.path.isfile(_tilt_matrix_path(data_path, img_size, Dr0))


def precompute_all_corr_matrices(data_path: str) -> None:
    """
    Ensure all four PSF correlation matrices are cached before any image
    is processed.  Only generates files that are not already on disk.
    """
    print(f"[images/setup] Pre-computing correlation matrices for {CORR_VALUES} …")
    for corr in CORR_VALUES:
        if _corr_matrix_exists(data_path, corr):
            print(f"  R-corr_{corr}.npy  ✓ cached")
        else:
            print(f"  Generating R-corr_{corr}.npy  (~10 min first time) …")
            corr_mat(corr, data_path)
            print(f"  R-corr_{corr}.npy  done.")
    print("[images/setup] All corr matrices ready.\n")


def _ensure_tilt_matrix(data_path: str, img_size: int, Dr0: float,
                         L: float) -> None:
    """Generate the tilt matrix for (img_size, Dr0) if not cached."""
    if not _tilt_matrix_exists(data_path, img_size, Dr0):
        r0 = D_APERTURE / Dr0
        tilt_mat(img_size, D_APERTURE, r0, L, data_path)


# ─────────────────────────────────────────────────────────────────────────────
# Live simulator parameter swap (mutate tensors in-place — no rebuild)
# ─────────────────────────────────────────────────────────────────────────────

def _swap_simulator_params(sim, Dr0: float, corr: float, scale: float,
                            data_path: str, device: torch.device) -> None:
    """
    Update the Simulator's internal tensors for new (Dr0, corr, scale)
    without rebuilding the whole object (avoids reloading P2S_model.pt
    and dictionary.npy on every image).
    """
    # PSF correlation matrix
    R_arr = np.load(os.path.join(data_path, f"R-corr_{corr}.npy"))
    sim.R = torch.tensor(R_arr, dtype=torch.float32, device=device)

    # Tilt matrix
    tilt_file = _tilt_matrix_path(data_path, sim.img_size, Dr0)
    d = np.load(tilt_file, allow_pickle=True)
    sim.S_half = torch.tensor(d.item()["s_half"], dtype=torch.float32, device=device)
    sim.const  = d.item()["const"]

    # Scalar params
    sim.Dr0   = torch.tensor(Dr0,   dtype=torch.float32, device=device)
    sim.scale = scale


# ─────────────────────────────────────────────────────────────────────────────
# Old single-combo precompute (still used by main() for video pipeline setup)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_matrices(img_size: int, D_over_r0: float, L: float,
                        corr: float, data_path: str) -> None:
    """Generate tilt and PSF matrices for a single combo if not cached."""
    D  = D_APERTURE
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

    - img_size fixed at IMAGE_IMG_SIZE (1024) regardless of config.yaml.
    - Each image receives a freshly sampled random parameter set:
        D_over_r0 ~ Uniform[0.1, 3.0]
        L         ~ Uniform[500, 5000]
        corr      ~ choice([-0.1, 0.0, 0.5, 0.9])
        scale     ~ Uniform[0.5, 2.0]
    - A .json label file is written next to every output image.
    - A master results_summary.csv with all parameters is written to
      output_path/images/.
    """
    from simulator import Simulator

    det_path = Path(cfg.get("dataset_det_path", "./dataset/visdrone_det_val"))
    img_out  = output_path / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # Locate image files — try det_path/images/ first, then root
    search_dir = det_path / "images" if (det_path / "images").exists() else det_path
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(p for p in search_dir.iterdir() if p.suffix.lower() in exts)

    if not image_files:
        print(f"[images] No images found in {search_dir} — skipping image batch.")
        return []

    print(f"[images] Found {len(image_files)} images in {search_dir}")
    print(f"[images] Using fixed img_size={IMAGE_IMG_SIZE} for all images")

    data_path  = str(Path(cfg["data_path"]))
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[images] CUDA not available — falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    img_size = IMAGE_IMG_SIZE

    # ── Step 1: Pre-compute ALL corr matrices before touching any image ───────
    precompute_all_corr_matrices(data_path)

    # ── Step 2: Build initial simulator with arbitrary starting params ────────
    init_Dr0  = 1.0
    init_corr = CORR_VALUES[0]
    init_scale = 1.0

    # Ensure tilt matrix exists for initial Dr0
    _ensure_tilt_matrix(data_path, img_size, init_Dr0, L=3000.0)

    simulator = Simulator(
        Dr0=init_Dr0,
        img_size=img_size,
        corr=init_corr,
        data_path=data_path,
        device=device_str,
        scale=init_scale,
    ).to(device, dtype=torch.float32)
    simulator.eval()

    # ── Step 3: Process each image with its own random params ─────────────────
    results = []
    t_start = time.time()

    csv_fields = [
        "source_image", "output_image", "label_file",
        "img_size", "D_over_r0", "r0", "L", "corr", "scale",
        "turbulence_strength", "time_s", "status",
    ]

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            t0 = time.time()

            # ── Sample random parameters for this image ───────────────────
            Dr0   = round(random.uniform(*DR0_RANGE), 4)
            L     = round(random.uniform(*L_RANGE), 2)
            corr  = random.choice(CORR_VALUES)
            scale = round(random.uniform(*SCALE_RANGE), 4)
            r0    = round(D_APERTURE / Dr0, 6)
            strength = strength_label(Dr0)

            # Ensure tilt matrix for this Dr0/L combo exists (generates if needed)
            _ensure_tilt_matrix(data_path, img_size, Dr0, L)

            # Swap simulator internals in-place
            _swap_simulator_params(simulator, Dr0, corr, scale, data_path, device)

            out_name   = img_path.stem + "_turbulence" + img_path.suffix
            label_name = img_path.stem + "_turbulence.json"
            out_file   = img_out / out_name
            label_file = img_out / label_name

            try:
                frame_bgr = cv2.imread(str(img_path))
                if frame_bgr is None:
                    raise IOError(f"Cannot read {img_path}")

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                src_h, src_w = frame_rgb.shape[:2]

                frame_sq = cv2.resize(frame_rgb, (img_size, img_size),
                                      interpolation=cv2.INTER_LINEAR)

                img_t = (torch.from_numpy(frame_sq)
                         .to(device, dtype=torch.float32)
                         .permute(2, 0, 1) / 255.0)

                out_t = simulator(img_t)

                out_np = out_t.cpu().numpy()
                if out_np.ndim == 3:
                    out_np = out_np.transpose(1, 2, 0)
                out_np    = np.clip(out_np, 0.0, 1.0)
                out_uint8 = (out_np * 255).astype(np.uint8)

                if (img_size, img_size) != (src_h, src_w):
                    out_uint8 = cv2.resize(out_uint8, (src_w, src_h),
                                           interpolation=cv2.INTER_LINEAR)

                out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_file), out_bgr)

                # ── Write per-image JSON label ────────────────────────────
                label = {
                    "source_image":        img_path.name,
                    "img_size":            img_size,
                    "D_over_r0":           Dr0,
                    "L":                   L,
                    "corr":                corr,
                    "scale":               scale,
                    "r0":                  r0,
                    "turbulence_strength": strength,
                }
                with open(label_file, "w") as lf:
                    json.dump(label, lf, indent=2)

                elapsed = time.time() - t0
                status  = "ok"

            except Exception as exc:
                elapsed   = time.time() - t0
                status    = f"error({exc})"
                out_name  = img_path.name
                label_name = ""

            results.append({
                "source_image":        img_path.name,
                "output_image":        out_name,
                "label_file":          label_name,
                "img_size":            img_size,
                "D_over_r0":           Dr0,
                "r0":                  r0,
                "L":                   L,
                "corr":                corr,
                "scale":               scale,
                "turbulence_strength": strength,
                "time_s":              f"{elapsed:.2f}",
                "status":              status,
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
                ok = sum(1 for r in results if r["status"] == "ok")
                print(f"  [{i+1}/{len(image_files)}] {ok} ok  "
                      f"D/r0={Dr0:.3f}  L={L:.0f}  corr={corr}  "
                      f"scale={scale:.2f}  ({strength})",
                      flush=True)

    total    = time.time() - t_start
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"[images] Done: {ok_count}/{len(results)} images in {total:.1f}s → {img_out}")

    # ── Master CSV with all parameters ────────────────────────────────────────
    csv_path = img_out / "results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
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
    Apply turbulence to ALL video sequences in the VID-val dataset.
    Sequences are discovered directly from dataset_path/sequences/ —
    no manifest file or subset selection involved.
    Results are written to  output_path/videos/
    """
    import shutil

    dataset_path = Path(cfg["dataset_path"])
    vid_out      = output_path / "videos"
    vid_out.mkdir(parents=True, exist_ok=True)

    seq_root = dataset_path / "sequences"
    if not seq_root.exists():
        print(f"[videos] Sequences directory not found: {seq_root}")
        print("         Make sure the VID-val dataset was extracted correctly.")
        sys.exit(1)

    all_clips = sorted(p.name for p in seq_root.iterdir() if p.is_dir())
    if not all_clips:
        print(f"[videos] No sequence folders found in {seq_root} — skipping.")
        return []

    print(f"[videos] Processing ALL {len(all_clips)} sequences → {vid_out}")

    scale   = float(cfg.get("scale", 1.0))
    D       = D_APERTURE
    min_sec = float(cfg.get("param_change_min_seconds", 1))
    max_sec = float(cfg.get("param_change_max_seconds", 30))

    tmp_dir = dataset_path.parent / "_tmp_videos"
    tmp_dir.mkdir(exist_ok=True)

    results = []
    t_start = time.time()

    for i, clip_name in enumerate(all_clips):
        seq_dir = dataset_path / "sequences" / clip_name

        if not seq_dir.exists():
            print(f"  [skip] Sequence folder not found: {seq_dir}")
            continue

        print(f"\n[{i+1}/{len(all_clips)}] {clip_name}")

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

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    total    = time.time() - t_start
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"\n[videos] Done: {ok_count}/{len(results)} clips in {total:.1f}s → {vid_out}")

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
    output_path = Path(cfg["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available — falling back to CPU.")
        device_str = "cpu"
        cfg["device"] = device_str
    print(f"[info] Device: {device_str}")
    print(f"[info] Image pipeline: img_size={IMAGE_IMG_SIZE}, random params per image")
    print(f"[info] Video pipeline: img_size=1024, continuously varying params")

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
    p = argparse.ArgumentParser(
        description="Headless batch turbulence inference (images + videos)"
    )
    p.add_argument("--config", default="config.yaml",
                   help="Path to config YAML (default: config.yaml)")
    p.add_argument("--images-only", action="store_true",
                   help="Only process the image dataset")
    p.add_argument("--videos-only", action="store_true",
                   help="Only process the video dataset")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    main(cfg, images_only=args.images_only, videos_only=args.videos_only)
