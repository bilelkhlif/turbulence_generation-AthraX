"""
Apply P2S atmospheric turbulence simulation to a video.

Reference:
  Z. Mao, N. Chimitt, S. H. Chan, "Accelerating Atmospheric Turbulence
  Simulation via Learned Phase-to-Space Transform", ICCV 2021.
  https://arxiv.org/abs/2107.11627

Design
------
One fixed preset per video clip — no mid-video parameter changes.
A fresh Simulator is built for each call (same pattern as demo.py).
img_size=1024 for all videos.

Usage
-----
    python apply_turbulence_to_video.py \\
        --input  <video> \\
        --output <video> \\
        --D      0.1     \\
        --r0     <r0>    \\
        --L      <L>     \\
        --corr   <corr>  \\
        --scale  1.0     \\
        --img-size 1024  \\
        --preset-name <name>
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from simulator import Simulator
from turbStats import tilt_mat


# ─────────────────────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────────────────────

def process_video(args) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device:      {device}")
    print(f"[info] Preset:      {args.preset_name}")
    print(f"[info] D/r0:        {args.D / args.r0:.4f}  "
          f"(D={args.D}  r0={args.r0})")
    print(f"[info] L:           {args.L} m")
    print(f"[info] corr:        {args.corr}")
    print(f"[info] scale:       {args.scale}")
    print(f"[info] img_size:    {args.img_size}")

    data_path = os.path.join(SCRIPT_DIR, "data")
    Dr0       = args.D / args.r0

    # ── Generate tilt matrix if not cached (same call as demo.py) ────────────
    tilt_mat(args.img_size, args.D, args.r0, args.L, data_path)

    # ── Build a fresh Simulator for this clip ─────────────────────────────────
    simulator = Simulator(
        Dr0=Dr0,
        img_size=args.img_size,
        corr=args.corr,
        data_path=data_path,
        device=str(device),
        scale=args.scale,
    ).to(device, dtype=torch.float32)
    simulator.eval()

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] Input:  {args.input}  "
          f"({src_w}×{src_h} @ {src_fps:.2f} fps, ~{total} frames)")

    # ── Open writer ───────────────────────────────────────────────────────────
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, src_fps, (src_w, src_h))
    if not out_writer.isOpened():
        raise IOError(f"Cannot open output video: {args.output}")
    print(f"[info] Output: {args.output}\n")

    img_size = args.img_size

    # ── Frame loop ────────────────────────────────────────────────────────────
    frame_idx = 0
    written   = 0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_sq  = cv2.resize(frame_rgb, (img_size, img_size),
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
            out_writer.write(out_bgr)
            written   += 1
            frame_idx += 1

            if frame_idx % 50 == 0:
                pct = (frame_idx / total * 100) if total > 0 else 0
                print(f"  {frame_idx}/{total} frames ({pct:.1f}%)",
                      flush=True)

    cap.release()
    out_writer.release()
    print(f"\n[done] Wrote {written} frames → {args.output}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Apply fixed-preset P2S atmospheric turbulence to a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",       required=True,  help="Input video path")
    p.add_argument("--output",      required=True,  help="Output video path")
    p.add_argument("--D",           type=float, default=0.1,
                   help="Aperture diameter (metres)")
    p.add_argument("--r0",          type=float, default=0.083,
                   help="Fried parameter (metres)")
    p.add_argument("--L",           type=float, default=3000.0,
                   help="Propagation distance (metres)")
    p.add_argument("--corr",        type=float, default=-0.1,
                   help="PSF correlation value: one of -0.1, 0.0, 0.5, 0.9")
    p.add_argument("--scale",       type=float, default=1.0,
                   help="Tilt strength multiplier")
    p.add_argument("--img-size",    dest="img_size", type=int, default=1024,
                   help="Simulator internal square resolution")
    p.add_argument("--preset-name", dest="preset_name", default="unknown",
                   help="Preset label for logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args)
