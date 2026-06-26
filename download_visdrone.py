"""
download_visdrone.py — Download VisDrone dataset assets.

  IMAGE SET (automatic):
    VisDrone2019-DET-val  — 548 images, ~70 MB
    Source: public Ultralytics mirror (no auth required)
    https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip

  VIDEO SET (manual upload required):
    VisDrone2019-VID-val  — 48 video sequences, ~1.49 GB
    The official Google Drive link is access-restricted and cannot be
    downloaded programmatically. Users must:
      1. Download the Task-2 valset from:
         https://github.com/VisDrone/VisDrone-Dataset
      2. Upload VisDrone2019-VID-val.zip to Google Drive in a folder
         named  "visdrone"
      3. Re-run:  bash run.sh

    This script will then pull it automatically via rclone from
      gdrive:visdrone/VisDrone2019-VID-val.zip

Usage
-----
    python download_visdrone.py [--det-dest DIR] [--vid-dest DIR]
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ── Public Ultralytics mirror for DET-val (no auth) ──────────────────────────
DET_URL  = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
DET_ZIP  = "VisDrone2019-DET-val.zip"

# ── Google Drive path for VID-val (user must upload) ─────────────────────────
VID_ZIP        = "VisDrone2019-VID-val.zip"
GDRIVE_REMOTE  = "gdrive:visdrone"
GDRIVE_ZIP_PATH = f"{GDRIVE_REMOTE}/{VID_ZIP}"


# ─────────────────────────────────────────────────────────────────────────────
# Image dataset (DET-val)
# ─────────────────────────────────────────────────────────────────────────────

def download_det_val(dest: Path) -> None:
    """Download and extract VisDrone2019-DET-val from the Ultralytics mirror."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / DET_ZIP

    # ── 1. Download ─────────────────────────────────────────────────────────
    if zip_path.exists():
        print(f"[images] Zip already present at {zip_path} — skipping download.")
    else:
        print("[images] Downloading VisDrone2019-DET-val (~70 MB) …")
        try:
            import urllib.request
            urllib.request.urlretrieve(DET_URL, str(zip_path), _progress_hook)
            print()  # newline after progress bar
        except Exception as exc:
            print(f"[images] urllib failed ({exc}), trying wget …")
            ret = subprocess.run(["wget", "-q", "--show-progress", DET_URL, "-O", str(zip_path)])
            if ret.returncode != 0:
                raise RuntimeError(f"Failed to download {DET_URL}")

    # ── 2. Extract ──────────────────────────────────────────────────────────
    images_dir = dest / "images"
    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"[images] Already extracted at {dest} — skipping.")
        return

    print(f"[images] Extracting {zip_path} to {dest} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    _normalise_layout(dest)

    img_count = sum(1 for p in dest.rglob("*.jpg")) + sum(1 for p in dest.rglob("*.png"))
    print(f"[images] ✓ Extracted ~{img_count} images to {dest}")


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        sys.stdout.write(f"\r  [{bar}] {pct}%  ({downloaded // 1024 // 1024} MB)")
        sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Video dataset (VID-val) — via rclone from user's Google Drive
# ─────────────────────────────────────────────────────────────────────────────

def download_vid_val(dest: Path) -> bool:
    """
    Try to pull VisDrone2019-VID-val.zip from  gdrive:visdrone/  via rclone.

    Returns True if successful, False if the file was not found in Drive.
    Raises RuntimeError if rclone is not installed.
    """
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / VID_ZIP

    # ── Check rclone availability ────────────────────────────────────────────
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone is not installed or not on PATH.\n"
            "  Install: https://rclone.org/install/\n"
            "  Then run: rclone config  (to authorise Google Drive)"
        )

    # ── Check if zip already downloaded ─────────────────────────────────────
    if zip_path.exists():
        print(f"[videos] Zip already present at {zip_path} — skipping rclone.")
    else:
        # Check whether the file actually exists in Google Drive
        print(f"[videos] Checking {GDRIVE_ZIP_PATH} …")
        check = subprocess.run(
            ["rclone", "lsf", GDRIVE_ZIP_PATH],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0 or VID_ZIP not in check.stdout:
            return False   # file not found → caller will print instructions

        # Pull the zip
        print(f"[videos] Pulling {VID_ZIP} from Google Drive via rclone …")
        ret = subprocess.run(
            ["rclone", "copy", GDRIVE_ZIP_PATH, str(dest), "--progress"],
        )
        if ret.returncode != 0:
            raise RuntimeError(f"rclone copy failed with exit code {ret.returncode}")

    # ── Extract ──────────────────────────────────────────────────────────────
    seq_dir = dest / "sequences"
    ann_dir = dest / "annotations"
    if seq_dir.exists() and ann_dir.exists() and any(seq_dir.iterdir()):
        print(f"[videos] Already extracted at {dest} — skipping.")
        return True

    print(f"[videos] Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    _normalise_layout(dest)

    seq_count = sum(1 for p in seq_dir.iterdir() if p.is_dir()) if seq_dir.exists() else 0
    print(f"[videos] ✓ Extracted {seq_count} sequences to {dest}")
    return True


def print_vid_instructions() -> None:
    """Print clear manual-upload instructions when VID-val is not in Drive."""
    print()
    print("  ┌──────────────────────────────────────────────────────────────────────┐")
    print("  │  VisDrone2019-VID-val NOT FOUND in gdrive:visdrone/                 │")
    print("  │                                                                      │")
    print("  │  The VID valset cannot be downloaded programmatically (restricted    │")
    print("  │  Google Drive). Please follow these steps:                           │")
    print("  │                                                                      │")
    print("  │  1. Go to: https://github.com/VisDrone/VisDrone-Dataset             │")
    print("  │     → Task 2 (Multi-Object Tracking) → Val → Google Drive link      │")
    print("  │     Download: VisDrone2019-VID-val.zip (~1.49 GB)                   │")
    print("  │                                                                      │")
    print("  │  2. Upload the zip to your Google Drive in a folder called:         │")
    print("  │       visdrone/                                                      │")
    print("  │     Final path:  gdrive:visdrone/VisDrone2019-VID-val.zip           │")
    print("  │                                                                      │")
    print("  │  3. Re-run:  bash run.sh                                            │")
    print("  │                                                                      │")
    print("  │  NOTE: If rclone is not yet configured for Google Drive, run:       │")
    print("  │    rclone config                                                     │")
    print("  │  and add a remote named  'gdrive'  pointing to your Google Drive.   │")
    print("  └──────────────────────────────────────────────────────────────────────┘")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Layout normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_layout(dest: Path) -> None:
    """
    Flatten a single-level nested directory, e.g.:
        dest/VisDrone2019-DET-val/images/...  →  dest/images/...
    """
    children = [p for p in dest.iterdir()]
    if (
        len(children) == 1
        and children[0].is_dir()
        and not (dest / "images").exists()
        and not (dest / "sequences").exists()
    ):
        nested = children[0]
        for item in list(nested.iterdir()):
            shutil.move(str(item), str(dest / item.name))
        nested.rmdir()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download VisDrone2019 DET-val (images) and VID-val (videos)"
    )
    p.add_argument(
        "--det-dest",
        default="./dataset/visdrone_det_val",
        help="Destination for image dataset (default: ./dataset/visdrone_det_val)",
    )
    p.add_argument(
        "--vid-dest",
        default="./dataset/visdrone_vid_val",
        help="Destination for video dataset (default: ./dataset/visdrone_vid_val)",
    )
    p.add_argument(
        "--images-only", action="store_true",
        help="Only download the image dataset (skip video)",
    )
    p.add_argument(
        "--videos-only", action="store_true",
        help="Only download the video dataset (skip images)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.videos_only:
        download_det_val(Path(args.det_dest))

    if not args.images_only:
        found = download_vid_val(Path(args.vid_dest))
        if not found:
            print_vid_instructions()
            sys.exit(1)
