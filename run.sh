#!/usr/bin/env bash
# =============================================================================
# run.sh — one-shot setup + inference for TurbulenceSimulatorPython
# Designed for a fresh Ubuntu / Vast.ai GPU instance.
#
# Pipeline (6 steps):
#   1. Install dependencies (PyTorch CUDA build auto-selected)
#   2. Verify model artefacts (P2S_model.pt, dictionary.npy)
#   3a. Download VisDrone2019-DET-val images automatically (~70 MB, public mirror)
#   3b. Pull VisDrone2019-VID-val videos via rclone from Google Drive
#       (requires user to upload zip to gdrive:visdrone/ first)
#   4. Select a diverse 30% clip subset using annotation metadata (videos only)
#   5a. Apply turbulence to images  → run_batch.py  (per-image)
#   5b. Apply turbulence to videos  → run_batch.py  (per-frame via apply_turbulence_to_video.py)
#   6. All outputs saved to output_path from config.yaml
#
# Usage:
#   bash run.sh [--config <path>] [--skip-install] [--skip-dataset] [--skip-select]
#               [--images-only] [--videos-only]
#
# Flags:
#   --config <path>   Path to config YAML              (default: config.yaml)
#   --skip-install    Skip pip install                  (env already set up)
#   --skip-dataset    Skip dataset download             (datasets already present)
#   --skip-select     Skip diversity selection          (manifest already exists)
#   --images-only     Only process the image pipeline
#   --videos-only     Only process the video pipeline
# =============================================================================

set -euo pipefail

CONFIG="config.yaml"
SKIP_INSTALL=false
SKIP_DATASET=false
SKIP_SELECT=false
IMAGES_ONLY=false
VIDEOS_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)        CONFIG="$2"; shift 2 ;;
    --skip-install)  SKIP_INSTALL=true; shift ;;
    --skip-dataset)  SKIP_DATASET=true; shift ;;
    --skip-select)   SKIP_SELECT=true; shift ;;
    --images-only)   IMAGES_ONLY=true; shift ;;
    --videos-only)   VIDEOS_ONLY=true; shift ;;
    *) echo "[warn] Unknown flag: $1"; shift ;;
  esac
done

# ── Helper: read a value from config.yaml ────────────────────────────────────
_cfg() {
  python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('$1', '$2'))
"
}

echo "============================================================"
echo "  TurbulenceSimulatorPython — run"
echo "  Config : $CONFIG"
echo "  $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Install dependencies
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[1/6] Dependencies"

if [ "$SKIP_INSTALL" = false ]; then
  if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "      Detected CUDA $CUDA_VER"
    MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    if [ "$MAJOR" -ge 12 ]; then
      TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
      TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi
    pip install --quiet torch==2.3.1 torchvision==0.18.1 --index-url "$TORCH_INDEX"
  else
    echo "      No GPU detected — installing CPU-only PyTorch."
    pip install --quiet torch==2.3.1 torchvision==0.18.1 \
      --index-url "https://download.pytorch.org/whl/cpu"
  fi
  pip install --quiet -r requirements.txt
  echo "      ✓ Dependencies installed."
else
  echo "      Skipped (--skip-install)."
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Verify model artefacts
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[2/6] Model artefacts"

DATA_PATH=$(_cfg "data_path" "./data")

if [ ! -f "$DATA_PATH/P2S_model.pt" ] || [ ! -f "$DATA_PATH/dictionary.npy" ]; then
  echo ""
  echo "  ┌──────────────────────────────────────────────────────────────────┐"
  echo "  │  MISSING REQUIRED MODEL FILES                                    │"
  echo "  │                                                                  │"
  echo "  │  Place these two files in  $DATA_PATH/                           │"
  echo "  │    • P2S_model.pt                                                │"
  echo "  │    • dictionary.npy                                              │"
  echo "  │                                                                  │"
  echo "  │  Get them from:                                                  │"
  echo "  │    https://github.com/Riponcs/TurbulenceSimulatorPython          │"
  echo "  │    (data/ folder in that repo)                                   │"
  echo "  │                                                                  │"
  echo "  │  Or: https://engineering.purdue.edu/ChanGroup/project_turbulence │"
  echo "  └──────────────────────────────────────────────────────────────────┘"
  exit 1
fi
echo "      P2S_model.pt   ✓"
echo "      dictionary.npy ✓"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3a — Download VisDrone2019-DET-val images (automatic, public mirror)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[3a/6] VisDrone2019-DET-val images (automatic download)"

DET_DATASET_PATH=$(_cfg "dataset_det_path" "./dataset/visdrone_det_val")

if [ "$SKIP_DATASET" = false ] && [ "$VIDEOS_ONLY" = false ]; then
  python3 download_visdrone.py \
    --images-only \
    --det-dest "$DET_DATASET_PATH"
  echo "      ✓ Image dataset ready at $DET_DATASET_PATH"
else
  if [ "$VIDEOS_ONLY" = true ]; then
    echo "      Skipped (--videos-only)."
  else
    echo "      Skipped (--skip-dataset)."
    if [ ! -d "$DET_DATASET_PATH" ] || [ -z "$(ls -A "$DET_DATASET_PATH" 2>/dev/null)" ]; then
      echo "      [error] $DET_DATASET_PATH is missing or empty. Remove --skip-dataset."
      exit 1
    fi
    echo "      Using existing image dataset at $DET_DATASET_PATH"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3b — Pull VisDrone2019-VID-val videos via rclone (or instruct user)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[3b/6] VisDrone2019-VID-val videos (rclone from Google Drive)"

VID_DATASET_PATH=$(_cfg "dataset_path" "./dataset/visdrone_vid_val")
VID_ZIP_NAME="VisDrone2019-VID-val.zip"
GDRIVE_ZIP="gdrive:visdrone/${VID_ZIP_NAME}"

if [ "$SKIP_DATASET" = false ] && [ "$IMAGES_ONLY" = false ]; then

  if ! command -v rclone &>/dev/null; then
    echo ""
    echo "  ┌──────────────────────────────────────────────────────────────────────┐"
    echo "  │  rclone not found on PATH                                            │"
    echo "  │  Install it from: https://rclone.org/install/                        │"
    echo "  │  Then run:  rclone config  to authorise Google Drive                 │"
    echo "  └──────────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  Continuing with image pipeline only."
    IMAGES_ONLY=true
  else
    # Check whether the zip exists in gdrive:visdrone/
    echo "      Checking ${GDRIVE_ZIP} …"
    if rclone lsf "$GDRIVE_ZIP" &>/dev/null 2>&1; then
      echo "      Found — pulling via rclone …"
      mkdir -p "./dataset"
      rclone copy "$GDRIVE_ZIP" "./dataset/" --progress
      python3 download_visdrone.py \
        --videos-only \
        --vid-dest "$VID_DATASET_PATH"
      echo "      ✓ Video dataset ready at $VID_DATASET_PATH"
    else
      echo ""
      echo "  ┌──────────────────────────────────────────────────────────────────────┐"
      echo "  │  VisDrone2019-VID-val NOT FOUND in gdrive:visdrone/                 │"
      echo "  │                                                                      │"
      echo "  │  The VID valset cannot be downloaded programmatically (restricted    │"
      echo "  │  Google Drive). Please follow these steps:                           │"
      echo "  │                                                                      │"
      echo "  │  1. Go to: https://github.com/VisDrone/VisDrone-Dataset             │"
      echo "  │     → Task 2 (Multi-Object Tracking) → Val → Google Drive link      │"
      echo "  │     Download: VisDrone2019-VID-val.zip  (~1.49 GB)                  │"
      echo "  │                                                                      │"
      echo "  │  2. Upload the zip to your Google Drive in a folder called:         │"
      echo "  │       visdrone/                                                      │"
      echo "  │     Final path in Drive:  gdrive:visdrone/VisDrone2019-VID-val.zip  │"
      echo "  │                                                                      │"
      echo "  │  3. Re-run:  bash run.sh                                            │"
      echo "  │                                                                      │"
      echo "  │  NOTE: If rclone is not yet configured for Google Drive, run:       │"
      echo "  │    rclone config                                                     │"
      echo "  │  and add a remote named 'gdrive' pointing to your Google Drive.     │"
      echo "  └──────────────────────────────────────────────────────────────────────┘"
      echo ""
      if [ "$IMAGES_ONLY" = false ]; then
        echo "  Continuing with image pipeline only."
        IMAGES_ONLY=true
      fi
    fi
  fi

else
  if [ "$IMAGES_ONLY" = true ]; then
    echo "      Skipped (--images-only)."
  else
    echo "      Skipped (--skip-dataset)."
    SEQ_DIR="$VID_DATASET_PATH/sequences"
    if [ ! -d "$SEQ_DIR" ] || [ -z "$(ls -A "$SEQ_DIR" 2>/dev/null)" ]; then
      echo "      [error] $SEQ_DIR is missing or empty. Remove --skip-dataset."
      exit 1
    fi
    echo "      Using existing video dataset at $VID_DATASET_PATH"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Diversity-based clip selection (videos only)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[4/6] Clip selection (video sequences)"

MANIFEST="$(dirname "$VID_DATASET_PATH")/selected_clips.json"
FRACTION=$(_cfg "subset_fraction" "0.30")

if [ "$IMAGES_ONLY" = true ]; then
  echo "      Skipped (images-only mode)."
elif [ "$SKIP_SELECT" = false ]; then
  python3 select_diverse_clips.py \
    --dataset  "$VID_DATASET_PATH" \
    --fraction "$FRACTION" \
    --out      "$MANIFEST"
  echo "      ✓ Manifest written to $MANIFEST"
else
  echo "      Skipped (--skip-select)."
  if [ ! -f "$MANIFEST" ]; then
    echo "      [error] Manifest $MANIFEST not found. Remove --skip-select."
    exit 1
  fi
  echo "      Using existing manifest: $MANIFEST"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Batch turbulence inference (images + videos)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[5/6] Batch turbulence inference"

OUTPUT_PATH=$(_cfg "output_path" "./outputs")
mkdir -p "$OUTPUT_PATH"

BATCH_ARGS="--config $CONFIG"
if [ "$IMAGES_ONLY" = true ]; then
  BATCH_ARGS="$BATCH_ARGS --images-only"
elif [ "$VIDEOS_ONLY" = true ]; then
  BATCH_ARGS="$BATCH_ARGS --videos-only"
fi

python3 run_batch.py $BATCH_ARGS

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Done
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[6/6] Complete"
echo ""
echo "============================================================"
echo "  Done.  Outputs saved to: $OUTPUT_PATH"
echo "    images/   — turbulence-degraded frames from DET-val"
echo "    videos/   — turbulence-degraded clips from VID-val"
echo "  $(date)"
echo "============================================================"
