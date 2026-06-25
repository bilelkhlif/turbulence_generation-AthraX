"""
Apply P2S atmospheric turbulence simulation to a video.

Reference:
  Z. Mao, N. Chimitt, S. H. Chan, "Accelerating Atmospheric Turbulence
  Simulation via Learned Phase-to-Space Transform", ICCV 2021.
  https://arxiv.org/abs/2107.11627

Usage:
  python apply_turbulence_to_video.py --input <video> --output <video> [options]

First-time setup:
  The script pre-computes two correlation matrices the first time it runs
  for a given combination of (img_size, D, r0, L, corr).  This can take
  several minutes.  On subsequent runs with the same parameters the cached
  .npy files in ./data are reused automatically.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

# ── make sure we can import from the repo root ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from simulator import Simulator
from turbStats import tilt_mat, corr_mat


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tilt_matrix_exists(data_path: str, img_size: int, Dr0: float) -> bool:
    fname = os.path.join(data_path, f"S_half-size_{img_size}-D_r0_{Dr0:.4f}.npy")
    return os.path.isfile(fname)


def _corr_matrix_exists(data_path: str, corr: float) -> bool:
    fname = os.path.join(data_path, f"R-corr_{corr}.npy")
    return os.path.isfile(fname)


def precompute_matrices(img_size: int, D: float, r0: float, L: float,
                        corr: float, data_path: str) -> None:
    """Generate the two required .npy files if they are not already cached."""
    Dr0 = D / r0

    if not _corr_matrix_exists(data_path, corr):
        print(f"[setup] Generating PSF correlation matrix R-corr_{corr}.npy  "
              f"(this can take ~10 min) …")
        corr_mat(corr, data_path)
        print("[setup] Done.")
    else:
        print(f"[setup] PSF correlation matrix already exists — skipping.")

    if not _tilt_matrix_exists(data_path, img_size, Dr0):
        print(f"[setup] Generating tilt matrix for size={img_size}, D/r0={Dr0:.4f}  "
              f"(this can take a few minutes) …")
        tilt_mat(img_size, D, r0, L, data_path)
        print("[setup] Done.")
    else:
        print(f"[setup] Tilt matrix already exists — skipping.")


# ─────────────────────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────────────────────

def process_video(args) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")

    data_path = os.path.join(SCRIPT_DIR, "data")

    # 1. Pre-compute correlation matrices (no-op if cached) ──────────────────
    precompute_matrices(
        img_size=args.img_size,
        D=args.D,
        r0=args.r0,
        L=args.L,
        corr=args.corr,
        data_path=data_path,
    )

    # 2. Build simulator ──────────────────────────────────────────────────────
    Dr0 = args.D / args.r0
    print(f"[info] D/r0 = {Dr0:.4f}  (turbulence strength dial)")
    simulator = Simulator(
        Dr0=Dr0,
        img_size=args.img_size,
        corr=args.corr,
        data_path=data_path,
        device=str(device),
        scale=args.scale,
    ).to(device, dtype=torch.float32)
    simulator.eval()

    # 3. Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input}")

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] Input:  {args.input}  ({src_w}×{src_h} @ {src_fps:.2f} fps, ~{total} frames)")

    # 4. Open writer (output keeps original resolution and fps) ───────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, src_fps, (src_w, src_h))
    if not out_writer.isOpened():
        raise IOError(f"Cannot open output video: {args.output}")
    print(f"[info] Output: {args.output}")

    sim_size = args.img_size  # square size used by the simulator

    # 5. Frame loop ───────────────────────────────────────────────────────────
    frame_idx = 0
    written   = 0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # ── resize to simulator's square input ────────────────────────
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_sq  = cv2.resize(frame_rgb, (sim_size, sim_size),
                                   interpolation=cv2.INTER_LINEAR)

            # ── BGR uint8 → float32 tensor (C, H, W) in [0, 1] ───────────
            img_t = (torch.from_numpy(frame_sq).to(device, dtype=torch.float32)
                     .permute(2, 0, 1) / 255.0)

            # ── simulate  (Simulator.forward handles color via -1 channels)
            # The simulator flattens all channels: it views input as
            # (C, H, W) and internally does view((-1,1,H,W)) so all three
            # colour planes are processed together.
            out_t = simulator(img_t)

            # ── tensor → numpy uint8 ──────────────────────────────────────
            out_np = out_t.cpu().numpy()           # (C, H, W)  or (H, W)
            if out_np.ndim == 3:
                out_np = out_np.transpose(1, 2, 0) # → (H, W, C)
            out_np = np.clip(out_np, 0.0, 1.0)
            out_uint8 = (out_np * 255).astype(np.uint8)

            # ── resize back to original frame size ────────────────────────
            if (sim_size, sim_size) != (src_h, src_w):
                out_uint8 = cv2.resize(out_uint8, (src_w, src_h),
                                       interpolation=cv2.INTER_LINEAR)

            out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
            out_writer.write(out_bgr)
            written    += 1
            frame_idx  += 1

            if frame_idx % 50 == 0:
                pct = (frame_idx / total * 100) if total > 0 else 0
                print(f"  {frame_idx}/{total} frames  ({pct:.1f}%)", flush=True)

    cap.release()
    out_writer.release()
    print(f"[done] Wrote {written} frames -> {args.output}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Add P2S atmospheric turbulence to a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",    required=True,  help="Path to input video file")
    p.add_argument("--output",   required=True,  help="Path to output video file")

    # Turbulence physical parameters
    p.add_argument("--D",     type=float, default=0.1,
                   help="Aperture diameter (metres)")
    p.add_argument("--r0",    type=float, default=0.05,
                   help="Fried parameter (metres). Smaller = stronger turbulence")
    p.add_argument("--L",     type=float, default=3000.0,
                   help="Propagation distance (metres)")

    # Simulator hyper-parameters
    p.add_argument("--img_size", type=int,   default=512,
                   help="Square size fed to the simulator {128,256,512,1024}")
    p.add_argument("--corr",     type=float, default=-0.1,
                   help="PSF temporal correlation strength {-5 … -0.01}")
    p.add_argument("--scale",    type=float, default=1.0,
                   help="Artificial tilt strength multiplier")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args)
