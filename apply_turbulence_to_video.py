"""
Apply P2S atmospheric turbulence simulation to a video with continuously
varying parameters.

Reference:
  Z. Mao, N. Chimitt, S. H. Chan, "Accelerating Atmospheric Turbulence
  Simulation via Learned Phase-to-Space Transform", ICCV 2021.
  https://arxiv.org/abs/2107.11627

Key features
------------
* video_img_size=1024 is hard-coded for all video processing regardless of
  what config.yaml says — gives the highest quality simulation.
* Turbulence parameters (D_over_r0, L, corr) vary continuously across frames.
  - The interval between parameter changes is random: uniformly drawn from
    [param_change_min_seconds, param_change_max_seconds] each time.
  - Actual FPS is read from the video file so seconds → frames conversion
    is always correct.
  - Transitions use cosine interpolation so there are no jarring jumps.
  - corr is chosen from the discrete set [-0.1, 0.0, 0.5, 0.9].
* ALL tilt and correlation matrices for every corr value AND a grid of
  D_over_r0 × L combinations are pre-computed before processing begins —
  zero pauses mid-video.
* Current parameters and frames-until-next-change are printed every 50 frames.

Usage
-----
    python apply_turbulence_to_video.py --input <video> --output <video> [options]

CLI options
-----------
  --input           Path to input video (required)
  --output          Path to output video (required)
  --D               Aperture diameter in metres          (default 0.1)
  --scale           Tilt strength multiplier             (default 1.0)
  --param-change-min-seconds   Min seconds between changes  (default 1)
  --param-change-max-seconds   Max seconds between changes  (default 30)
  --data-path       Override ./data directory

All other turbulence parameters (D_over_r0, L, corr) are varied automatically.
"""

import argparse
import math
import os
import random
import sys

import cv2
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from simulator import Simulator
from turbStats import tilt_mat, corr_mat

# ── Fixed video resolution ────────────────────────────────────────────────────
VIDEO_IMG_SIZE = 1024   # always use 1024 for video — overrides config.yaml

# ── Parameter variation ranges ────────────────────────────────────────────────
DR0_RANGE   = (0.1, 3.0)          # D/r0  (turbulence strength)
L_RANGE     = (500.0, 5000.0)     # propagation distance in metres
CORR_VALUES = [-0.1, 0.0, 0.5, 0.9]  # discrete PSF correlation choices

# ── Pre-computation grid: D_over_r0 × L combos computed ahead of time ────────
# We sample the range in steps so every possible random value is already cached.
DR0_PRECOMPUTE = [round(v, 4) for v in np.arange(0.1, 3.05, 0.1)]   # 0.1 … 3.0
L_PRECOMPUTE   = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0,
                  3500.0, 4000.0, 4500.0, 5000.0]


# ─────────────────────────────────────────────────────────────────────────────
# Matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def _corr_matrix_path(data_path: str, corr: float) -> str:
    return os.path.join(data_path, f"R-corr_{corr}.npy")


def _tilt_matrix_path(data_path: str, img_size: int, Dr0: float) -> str:
    return os.path.join(data_path, f"S_half-size_{img_size}-D_r0_{Dr0:.4f}.npy")


def precompute_all_matrices(data_path: str, img_size: int, D: float) -> None:
    """
    Pre-compute and cache:
      • One corr matrix per value in CORR_VALUES
      • One tilt matrix per (img_size, Dr0, L) combination in the grid
    Only missing files are generated; existing ones are skipped instantly.
    """
    total_corr = len(CORR_VALUES)
    total_tilt = len(DR0_PRECOMPUTE) * len(L_PRECOMPUTE)
    print(f"[precompute] Checking {total_corr} corr matrices and "
          f"{total_tilt} tilt matrices …")

    # ── Correlation (PSF) matrices ───────────────────────────────────────────
    for i, corr in enumerate(CORR_VALUES):
        path = _corr_matrix_path(data_path, corr)
        if os.path.isfile(path):
            print(f"  [corr {i+1}/{total_corr}] R-corr_{corr}.npy  ✓ cached")
        else:
            print(f"  [corr {i+1}/{total_corr}] Generating R-corr_{corr}.npy "
                  f"(~10 min first time) …")
            corr_mat(corr, data_path)
            print(f"  [corr {i+1}/{total_corr}] Done.")

    # ── Tilt matrices ────────────────────────────────────────────────────────
    idx = 0
    for Dr0 in DR0_PRECOMPUTE:
        for L in L_PRECOMPUTE:
            idx += 1
            path = _tilt_matrix_path(data_path, img_size, Dr0)
            if os.path.isfile(path):
                print(f"  [tilt {idx}/{total_tilt}] "
                      f"size={img_size} Dr0={Dr0:.2f} L={L:.0f}  ✓ cached")
            else:
                r0 = D / Dr0
                print(f"  [tilt {idx}/{total_tilt}] "
                      f"Generating size={img_size} Dr0={Dr0:.2f} L={L:.0f} …")
                tilt_mat(img_size, D, r0, L, data_path)
                print(f"  [tilt {idx}/{total_tilt}] Done.")

    print("[precompute] All matrices ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Cosine interpolation
# ─────────────────────────────────────────────────────────────────────────────

def cosine_interp(a: float, b: float, t: float) -> float:
    """Smoothly interpolate from a→b using a cosine curve; t in [0,1]."""
    mu = (1.0 - math.cos(t * math.pi)) / 2.0
    return a * (1.0 - mu) + b * mu


# ─────────────────────────────────────────────────────────────────────────────
# Live parameter swap — mutate Simulator tensors in-place (no rebuild)
# ─────────────────────────────────────────────────────────────────────────────

def _load_corr_tensor(data_path: str, corr: float, device: torch.device) -> torch.Tensor:
    arr = np.load(_corr_matrix_path(data_path, corr))
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _load_tilt_data(data_path: str, img_size: int, Dr0: float):
    """Returns (s_half_array, const_scalar) from the cached .npy file."""
    d = np.load(_tilt_matrix_path(data_path, img_size, Dr0), allow_pickle=True)
    return d.item()["s_half"], d.item()["const"]


def apply_params_to_simulator(
    sim: Simulator,
    Dr0: float,
    corr: float,
    data_path: str,
    device: torch.device,
) -> None:
    """
    Swap the Simulator's internal tensors to reflect new (Dr0, corr) values.
    The tilt S_half depends only on Dr0 (L baked into the file at precompute time).
    corr controls self.R.
    """
    # Update Dr0 scalar
    sim.Dr0 = torch.tensor(Dr0, dtype=torch.float32, device=device)

    # Update PSF correlation matrix
    sim.R = _load_corr_tensor(data_path, corr, device)

    # Update tilt matrix
    s_half_arr, const = _load_tilt_data(data_path, sim.img_size, Dr0)
    sim.S_half = torch.tensor(s_half_arr, dtype=torch.float32, device=device)
    sim.const  = const


# ─────────────────────────────────────────────────────────────────────────────
# Parameter schedule: random intervals + cosine transitions
# ─────────────────────────────────────────────────────────────────────────────

class ParamScheduler:
    """
    Manages smoothly varying turbulence parameters over time.

    At each interval boundary a new random target is drawn and the scheduler
    cosine-interpolates toward it over the next interval.  The interval length
    is itself re-randomised at every boundary.
    """

    def __init__(
        self,
        fps: float,
        min_seconds: float,
        max_seconds: float,
        init_Dr0: float,
        init_L: float,
        init_corr: float,
    ):
        self.fps        = fps
        self.min_frames = max(1, int(round(min_seconds * fps)))
        self.max_frames = max(self.min_frames, int(round(max_seconds * fps)))

        # Current ("from") values
        self.cur_Dr0  = init_Dr0
        self.cur_L    = init_L
        self.cur_corr = init_corr

        # Target ("to") values — start same as current
        self.tgt_Dr0  = init_Dr0
        self.tgt_L    = init_L
        self.tgt_corr = init_corr

        # Interval state
        self.interval_frames  = self._new_interval()
        self.frames_in_segment = 0

    def _new_interval(self) -> int:
        return random.randint(self.min_frames, self.max_frames)

    def _new_targets(self):
        self.tgt_Dr0  = round(random.uniform(*DR0_RANGE), 4)
        self.tgt_L    = random.choice(L_PRECOMPUTE)
        self.tgt_corr = random.choice(CORR_VALUES)

    def step(self) -> tuple:
        """
        Advance one frame and return (Dr0, L, corr, frames_until_change).
        Dr0 / L are cosine-interpolated; corr switches at the boundary.
        """
        t = self.frames_in_segment / max(1, self.interval_frames)
        t = min(t, 1.0)

        Dr0  = cosine_interp(self.cur_Dr0, self.tgt_Dr0, t)
        L    = cosine_interp(self.cur_L,   self.tgt_L,   t)
        # corr is discrete — snap to target when we're past the midpoint
        corr = self.tgt_corr if t >= 0.5 else self.cur_corr

        frames_left = self.interval_frames - self.frames_in_segment

        self.frames_in_segment += 1
        if self.frames_in_segment >= self.interval_frames:
            # Commit current transition
            self.cur_Dr0  = self.tgt_Dr0
            self.cur_L    = self.tgt_L
            self.cur_corr = self.tgt_corr
            # Draw next interval and targets
            self.interval_frames   = self._new_interval()
            self.frames_in_segment = 0
            self._new_targets()
            print(
                f"  [param change] "
                f"→ D/r0={self.tgt_Dr0:.3f}  L={self.tgt_L:.0f}m  "
                f"corr={self.tgt_corr}  "
                f"(next change in {self.interval_frames} frames / "
                f"{self.interval_frames/self.fps:.1f}s)",
                flush=True,
            )

        return Dr0, L, corr, frames_left


# ─────────────────────────────────────────────────────────────────────────────
# Nearest precomputed Dr0 lookup
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_precomputed_Dr0(Dr0: float) -> float:
    """Return the closest value in DR0_PRECOMPUTE to the given Dr0."""
    return min(DR0_PRECOMPUTE, key=lambda x: abs(x - Dr0))


# ─────────────────────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────────────────────

def process_video(args) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")

    data_path = getattr(args, "data_path", None) or os.path.join(SCRIPT_DIR, "data")
    D         = args.D
    scale     = args.scale
    img_size  = VIDEO_IMG_SIZE  # always 1024 for video
    print(f"[info] Video img_size fixed at {img_size} (overrides config.yaml)")

    # ── Step 1: Pre-compute ALL matrices before touching any frame ────────────
    precompute_all_matrices(data_path, img_size, D)

    # ── Step 2: Open video and read actual FPS ────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] Input:  {args.input}  "
          f"({src_w}×{src_h} @ {src_fps:.2f} fps, ~{total} frames)")

    min_sec = getattr(args, "param_change_min_seconds", 1)
    max_sec = getattr(args, "param_change_max_seconds", 30)
    print(f"[info] Param change interval: {min_sec}–{max_sec} s  "
          f"({int(min_sec*src_fps)}–{int(max_sec*src_fps)} frames at {src_fps:.1f} fps)")

    # ── Step 3: Build initial simulator ──────────────────────────────────────
    init_Dr0  = round(random.uniform(*DR0_RANGE), 4)
    init_L    = random.choice(L_PRECOMPUTE)
    init_corr = random.choice(CORR_VALUES)
    init_Dr0_snapped = _nearest_precomputed_Dr0(init_Dr0)

    print(f"[info] Initial params: D/r0={init_Dr0:.3f}  L={init_L:.0f}m  corr={init_corr}")

    simulator = Simulator(
        Dr0=init_Dr0_snapped,
        img_size=img_size,
        corr=init_corr,
        data_path=data_path,
        device=str(device),
        scale=scale,
    ).to(device, dtype=torch.float32)
    simulator.eval()

    # ── Step 4: Parameter scheduler ──────────────────────────────────────────
    scheduler = ParamScheduler(
        fps=src_fps,
        min_seconds=min_sec,
        max_seconds=max_sec,
        init_Dr0=init_Dr0,
        init_L=init_L,
        init_corr=init_corr,
    )

    # ── Step 5: Open writer ───────────────────────────────────────────────────
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, src_fps, (src_w, src_h))
    if not out_writer.isOpened():
        raise IOError(f"Cannot open output video: {args.output}")
    print(f"[info] Output: {args.output}\n")

    # ── Step 6: Frame loop ────────────────────────────────────────────────────
    frame_idx   = 0
    written     = 0
    prev_corr   = init_corr
    prev_Dr0_s  = init_Dr0_snapped  # snapped Dr0 last loaded into simulator

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Get interpolated params for this frame
            Dr0, L, corr, frames_left = scheduler.step()

            # Snap Dr0 to nearest precomputed value for tilt-matrix lookup
            Dr0_snapped = _nearest_precomputed_Dr0(Dr0)

            # Only reload simulator internals when they actually change
            if Dr0_snapped != prev_Dr0_s or corr != prev_corr:
                apply_params_to_simulator(simulator, Dr0_snapped, corr, data_path, device)
                prev_Dr0_s = Dr0_snapped
                prev_corr  = corr

            # Update Dr0 tensor live (controls Zernike scaling continuously)
            simulator.Dr0 = torch.tensor(Dr0, dtype=torch.float32, device=device)

            # ── Resize to simulator square input ─────────────────────────
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_sq  = cv2.resize(frame_rgb, (img_size, img_size),
                                   interpolation=cv2.INTER_LINEAR)

            img_t = (torch.from_numpy(frame_sq)
                     .to(device, dtype=torch.float32)
                     .permute(2, 0, 1) / 255.0)

            out_t  = simulator(img_t)

            # ── Tensor → numpy uint8 ──────────────────────────────────────
            out_np = out_t.cpu().numpy()
            if out_np.ndim == 3:
                out_np = out_np.transpose(1, 2, 0)
            out_np    = np.clip(out_np, 0.0, 1.0)
            out_uint8 = (out_np * 255).astype(np.uint8)

            # ── Resize back to original resolution ───────────────────────
            if (img_size, img_size) != (src_h, src_w):
                out_uint8 = cv2.resize(out_uint8, (src_w, src_h),
                                       interpolation=cv2.INTER_LINEAR)

            out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
            out_writer.write(out_bgr)
            written   += 1
            frame_idx += 1

            if frame_idx % 50 == 0:
                pct = (frame_idx / total * 100) if total > 0 else 0
                print(
                    f"  frame {frame_idx}/{total} ({pct:.1f}%)  "
                    f"D/r0={Dr0:.3f}  L={L:.0f}m  corr={corr}  "
                    f"next change in {frames_left} frames",
                    flush=True,
                )

    cap.release()
    out_writer.release()
    print(f"\n[done] Wrote {written} frames → {args.output}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Add continuously-varying P2S atmospheric turbulence to a video. "
            f"Always uses img_size={VIDEO_IMG_SIZE} for video (quality override)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  required=True, help="Path to input video file")
    p.add_argument("--output", required=True, help="Path to output video file")

    p.add_argument("--D",     type=float, default=0.1,
                   help="Aperture diameter (metres)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Tilt strength multiplier")
    p.add_argument("--data-path", dest="data_path", default=None,
                   help="Override path to the data/ directory")

    p.add_argument("--param-change-min-seconds", dest="param_change_min_seconds",
                   type=float, default=1.0,
                   help="Minimum seconds between turbulence parameter changes")
    p.add_argument("--param-change-max-seconds", dest="param_change_max_seconds",
                   type=float, default=30.0,
                   help="Maximum seconds between turbulence parameter changes")

    # Legacy flags kept for backward compatibility with run_batch.py calls
    # (they are silently ignored — Dr0/L/corr are now varied automatically)
    p.add_argument("--r0",       type=float, default=0.05, help=argparse.SUPPRESS)
    p.add_argument("--L",        type=float, default=3000.0, help=argparse.SUPPRESS)
    p.add_argument("--img_size", type=int,   default=VIDEO_IMG_SIZE, help=argparse.SUPPRESS)
    p.add_argument("--corr",     type=float, default=-0.1, help=argparse.SUPPRESS)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args)
