"""
download_visdrone.py — Download the VisDrone2019-VID validation set (Task 2).

  Dataset : VisDrone2019-VID valset (object detection in videos)
  Size    : ~1.49 GB zip
  Source  : official VisDrone Google Drive
            https://github.com/VisDrone/VisDrone-Dataset
  Licence : academic / non-commercial (VisDrone terms)

The zip contains sequence sub-folders, each holding:
  sequences/<seq_name>/          ← JPEG frames
  annotations/<seq_name>.txt     ← per-frame bounding-box annotations

Both are kept so that select_diverse_clips.py can use the annotations
to drive the diversity-based subset selection.

Usage
-----
    python download_visdrone.py [--dest DIR]

Options
-------
  --dest DIR   Root folder where the dataset is placed.
               Default: ./dataset/visdrone_vid_val  (matches config.yaml)
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

# ── Google Drive file ID for VisDrone2019-VID valset (1.49 GB) ────────────
# Source: https://github.com/VisDrone/VisDrone-Dataset  (Task 2 → valset)
GDRIVE_FILE_ID = "1fMCMqGMHN_HeFHGS4gOlSNu7NbVSfKoQ"
ZIP_NAME = "VisDrone2019-VID-val.zip"


def _require_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[setup] gdown not found — installing…")
        os.system(f"{sys.executable} -m pip install -q gdown==5.2.0")


def download_and_extract(dest: Path) -> None:
    try:
        import gdown
    except ImportError:
        _require_gdown()
        import gdown

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / ZIP_NAME

    # ── 1. Download ─────────────────────────────────────────────────────────
    if zip_path.exists():
        print(f"[download] Zip already present at {zip_path} — skipping download.")
    else:
        print("[download] Downloading VisDrone2019-VID valset (~1.49 GB) from Google Drive…")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(zip_path), quiet=False)

    # ── 2. Extract ──────────────────────────────────────────────────────────
    # Only extract if dest doesn't already look populated
    seq_dir = dest / "sequences"
    ann_dir = dest / "annotations"
    if seq_dir.exists() and ann_dir.exists() and any(seq_dir.iterdir()):
        print(f"[extract] Dataset already extracted at {dest} — skipping.")
        return

    print(f"[extract] Extracting {zip_path} to {dest} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    # The zip may unpack into a nested sub-folder; flatten if needed.
    # Expected final structure:  dest/sequences/...  dest/annotations/...
    _normalise_layout(dest)

    seq_count = sum(1 for p in seq_dir.iterdir() if p.is_dir()) if seq_dir.exists() else 0
    print(f"[done] Extracted {seq_count} sequences to {dest}")


def _normalise_layout(dest: Path) -> None:
    """
    Handle the case where the zip extracts into a single nested directory,
    e.g.  dest/VisDrone2019-VID-val/sequences/...
    Moves contents one level up if that is the structure.
    """
    children = [p for p in dest.iterdir()]
    if (
        len(children) == 1
        and children[0].is_dir()
        and not (dest / "sequences").exists()
    ):
        nested = children[0]
        for item in list(nested.iterdir()):
            shutil.move(str(item), str(dest / item.name))
        nested.rmdir()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download VisDrone2019-VID valset (Task 2 — video sequences)"
    )
    p.add_argument(
        "--dest",
        default="./dataset/visdrone_vid_val",
        help="Destination folder (default: ./dataset/visdrone_vid_val)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _require_gdown()
    download_and_extract(Path(args.dest))
