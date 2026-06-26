"""
run_batch.py — headless batch inference entry point.

Reads all settings from config.yaml, then processes BOTH:

  • Images  (visdrone_det_val)  → all 548 images processed via the Simulator,
    cycling evenly through 5 preset turbulence levels.
    img_size=1024 for all images.
    A .json label + master results_summary.csv are written.
    Outputs → output_path/images/

  • Videos  (visdrone_vid_val)  → ALL sequences processed frame-by-frame via
    apply_turbulence_to_video.py, one preset per clip (cycling through presets).
    img_size=1024 for all videos.
    Outputs → output_path/videos/

Presets (realistic for aerial drone footage, D/r0 ≤ 1.5):
    negligible  D/r0=0.2  L=500   corr=-0.1  scale=1.0
    weak        D/r0=0.5  L=1000  corr=-0.1  scale=1.0
    mild        D/r0=0.8  L=2000  corr=-0.1  scale=1.0
    moderate    D/r0=1.2  L=3000  corr=-0.1  scale=1.0
    strong      D/r0=1.5  L=3000  corr=0.5   scale=1.0

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

# ── Fixed simulation resolution for both pipelines ───────────────────────────
IMG_SIZE = 1024

# ── D (aperture diameter) fixed at 0.1 m throughout ─────────────────────────
D_APERTURE = 0.1

# ── Turbulence presets — realistic for aerial drone footage ──────────────────
# (name, D_over_r0, L, corr, scale)
# D/r0 never exceeds 1.5 — stays within the network's reliable range.
PRESETS = [
    ("negligible", 0.2,  500,  -0.1, 1.0),
    ("weak",       0.5, 1000,  -0.1, 1.0),
    ("mild",       0.8, 2000,  -0.1, 1.0),
    ("moderate",   1.2, 3000,  -0.1, 1.0),
    ("strong",     1.5, 3000,   0.5, 1.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Image batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_images(cfg: dict, output_path: Path) -> list:
    """
    Apply turbulence to every image in the DET-val dataset.

    Presets are cycled evenly across all images (image i gets
    PRESETS[i % len(PRESETS)]).  A fresh Simulator is built for each
    distinct preset — no tensor swapping, no bulk precomputation.

    Outputs:
      • output_path/images/<name>_turbulence.<ext>   — degraded image
      • output_path/images/<name>_turbulence.json    — per-image label
      • output_path/images/results_summary.csv       — master CSV
    """
    from simulator import Simulator
    from turbStats import tilt_mat

    det_path = Path(cfg.get("dataset_det_path", "./dataset/visdrone_det_val"))
    img_out  = output_path / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    search_dir = det_path / "images" if (det_path / "images").exists() else det_path
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(p for p in search_dir.iterdir() if p.suffix.lower() in exts)

    if not image_files:
        print(f"[images] No images found in {search_dir} — skipping.")
        return []

    print(f"[images] Found {len(image_files)} images in {search_dir}")
    print(f"[images] img_size={IMG_SIZE}  cycling through {len(PRESETS)} presets")

    data_path  = str(Path(cfg["data_path"]))
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[images] CUDA not available — falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # Build one Simulator per distinct preset and cache them.
    # tilt_mat is called inside Simulator.__init__ via the .npy file;
    # we pre-generate the file here exactly as demo.py does (once per preset).
    print("[images] Building simulators for all presets …")
    simulators = {}
    for preset_name, Dr0, L, corr, scale in PRESETS:
        r0 = D_APERTURE / Dr0
        print(f"  preset='{preset_name}'  D/r0={Dr0}  L={L}  corr={corr}  scale={scale}")
        # Generate tilt matrix if not already cached (same call as demo.py)
        tilt_mat(IMG_SIZE, D_APERTURE, r0, L, data_path)
        sim = Simulator(
            Dr0=Dr0,
            img_size=IMG_SIZE,
            corr=corr,
            data_path=data_path,
            device=device_str,
            scale=scale,
        ).to(device, dtype=torch.float32)
        sim.eval()
        simulators[preset_name] = sim
    print("[images] All simulators ready.\n")

    csv_fields = [
        "source_image", "output_image", "label_file",
        "preset", "img_size", "D_over_r0", "r0", "L", "corr", "scale",
        "time_s", "status",
    ]
    results = []
    t_start = time.time()

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            # Cycle through presets evenly
            preset_name, Dr0, L, corr, scale = PRESETS[i % len(PRESETS)]
            r0 = round(D_APERTURE / Dr0, 6)
            simulator = simulators[preset_name]

            out_name   = img_path.stem + "_turbulence" + img_path.suffix
            label_name = img_path.stem + "_turbulence.json"
            out_file   = img_out / out_name
            label_file = img_out / label_name

            t0 = time.time()
            try:
                frame_bgr = cv2.imread(str(img_path))
                if frame_bgr is None:
                    raise IOError(f"Cannot read {img_path}")

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                src_h, src_w = frame_rgb.shape[:2]

                frame_sq = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE),
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

                if (IMG_SIZE, IMG_SIZE) != (src_h, src_w):
                    out_uint8 = cv2.resize(out_uint8, (src_w, src_h),
                                           interpolation=cv2.INTER_LINEAR)

                out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_file), out_bgr)

                label = {
                    "source_image": img_path.name,
                    "preset":       preset_name,
                    "img_size":     IMG_SIZE,
                    "D_over_r0":    Dr0,
                    "L":            float(L),
                    "corr":         corr,
                    "scale":        scale,
                    "r0":           r0,
                }
                with open(label_file, "w") as lf:
                    json.dump(label, lf, indent=2)

                elapsed = time.time() - t0
                status  = "ok"

            except Exception as exc:
                elapsed    = time.time() - t0
                status     = f"error({exc})"
                out_name   = img_path.name
                label_name = ""

            results.append({
                "source_image": img_path.name,
                "output_image": out_name,
                "label_file":   label_name,
                "preset":       preset_name,
                "img_size":     IMG_SIZE,
                "D_over_r0":    Dr0,
                "r0":           r0,
                "L":            float(L),
                "corr":         corr,
                "scale":        scale,
                "time_s":       f"{elapsed:.2f}",
                "status":       status,
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
                ok = sum(1 for r in results if r["status"] == "ok")
                print(f"  [{i+1}/{len(image_files)}] {ok} ok  "
                      f"preset='{preset_name}'  D/r0={Dr0}  L={L}  corr={corr}",
                      flush=True)

    total    = time.time() - t_start
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"[images] Done: {ok_count}/{len(results)} images in {total:.1f}s → {img_out}")

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
    Each clip gets one preset (cycling through PRESETS).
    No mid-video parameter changes.
    Outputs → output_path/videos/
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
    print(f"[videos] One preset per clip, cycling through {len(PRESETS)} presets")

    tmp_dir = dataset_path.parent / "_tmp_videos"
    tmp_dir.mkdir(exist_ok=True)

    results = []
    t_start = time.time()

    for i, clip_name in enumerate(all_clips):
        seq_dir = dataset_path / "sequences" / clip_name
        if not seq_dir.exists():
            print(f"  [skip] Sequence folder not found: {seq_dir}")
            continue

        # Assign preset by cycling
        preset_name, Dr0, L, corr, scale = PRESETS[i % len(PRESETS)]
        r0 = D_APERTURE / Dr0

        print(f"\n[{i+1}/{len(all_clips)}] {clip_name}  "
              f"preset='{preset_name}'  D/r0={Dr0}  L={L}  corr={corr}")

        tmp_input = tmp_dir / f"{clip_name}.mp4"
        if not tmp_input.exists():
            print(f"  Assembling frames from {seq_dir} …")
            ok = frames_to_video(seq_dir, tmp_input, fps=30.0)
            if not ok:
                print(f"  [skip] No frames found in {seq_dir}")
                continue
        else:
            print(f"  Using existing temp video: {tmp_input}")

        out_video = vid_out / f"{clip_name}_{preset_name}_turbulence.mp4"

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "apply_turbulence_to_video.py"),
            "--input",      str(tmp_input),
            "--output",     str(out_video),
            "--D",          str(D_APERTURE),
            "--r0",         str(r0),
            "--L",          str(L),
            "--corr",       str(corr),
            "--scale",      str(scale),
            "--img-size",   str(IMG_SIZE),
            "--preset-name", preset_name,
        ]

        t0 = time.time()
        print(f"  Running: {' '.join(cmd)}", flush=True)
        ret = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
        elapsed = time.time() - t0

        status = "ok" if ret.returncode == 0 else f"error({ret.returncode})"
        results.append({
            "clip":    clip_name,
            "preset":  preset_name,
            "D_over_r0": Dr0,
            "L":       float(L),
            "corr":    corr,
            "scale":   scale,
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
        writer = csv.DictWriter(
            f, fieldnames=["clip", "preset", "D_over_r0", "L", "corr",
                           "scale", "output", "time_s", "status"])
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
        print("[warn] CUDA not available — falling back to CPU.")
        device_str = "cpu"
        cfg["device"] = device_str
    print(f"[info] Device: {device_str}")
    print(f"[info] img_size={IMG_SIZE} for both images and videos")
    print(f"[info] {len(PRESETS)} presets: "
          + "  ".join(f"{p[0]}(D/r0={p[1]})" for p in PRESETS))

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
