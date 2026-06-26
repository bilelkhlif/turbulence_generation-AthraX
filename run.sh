#!/usr/bin/env bash
# =============================================================================
# run.sh — one-shot setup + inference for TurbulenceSimulatorPython
# Designed for a fresh Ubuntu / Vast.ai GPU instance.
#
# Pipeline (5 steps):
#   1. Install dependencies (PyTorch CUDA build auto-selected)
#   2. Verify model artefacts (P2S_model.pt, dictionary.npy)
#   3. Download VisDrone2019-VID valset (~1.49 GB)
#   4. Select a diverse 30% clip subset using annotation metadata
#   5. Apply turbulence to each selected clip; save to output_path
#
# Usage:
#   bash run.sh [--config <path>] [--skip-install] [--skip-dataset] [--skip-select]
#
# Flags:
#   --config <path>   Path to config YAML          (default: config.yaml)
#   --skip-install    Skip pip install              (env already set up)
#   --skip-dataset    Skip VisDrone download        (dataset already present)
#   --skip-select     Skip diversity selection      (manifest already exists)
# =============================================================================

set -euo pipefail

CONFIG="config.yaml"
SKIP_INSTALL=false
SKIP_DATASET=false
SKIP_SELECT=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)        CONFIG="$2"; shift 2 ;;
    --skip-install)  SKIP_INSTALL=true; shift ;;
    --skip-dataset)  SKIP_DATASET=true; shift ;;
    --skip-select)   SKIP_SELECT=true; shift ;;
    *) echo "[warn] Unknown flag: $1"; shift ;;
  esac
done

# ── Helper: read a value from config.yaml without a full YAML library ─────────
_cfg() {
  python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('$1', '$2'))
"
}

echo "============================================================"
echo "  TurbulenceSimulatorPython — Vast.ai run"
echo "  Config : $CONFIG"
echo "  $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Install dependencies
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[1/5] Dependencies"

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
echo "[2/5] Model artefacts"

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
# STEP 3 — Download VisDrone-VID valset
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[3/5] VisDrone-VID valset"

DATASET_PATH=$(_cfg "dataset_path" "./dataset/visdrone_vid_val")

if [ "$SKIP_DATASET" = false ]; then
  python3 download_visdrone.py --dest "$DATASET_PATH"
  echo "      ✓ Dataset ready at $DATASET_PATH"
else
  echo "      Skipped (--skip-dataset)."
  SEQ_DIR="$DATASET_PATH/sequences"
  if [ ! -d "$SEQ_DIR" ] || [ -z "$(ls -A "$SEQ_DIR" 2>/dev/null)" ]; then
    echo "      [error] $SEQ_DIR is missing or empty. Remove --skip-dataset."
    exit 1
  fi
  echo "      Using existing dataset at $DATASET_PATH"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Diversity-based clip selection
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[4/5] Clip selection"

MANIFEST="$(dirname "$DATASET_PATH")/selected_clips.json"
FRACTION=$(_cfg "subset_fraction" "0.30")

if [ "$SKIP_SELECT" = false ]; then
  python3 select_diverse_clips.py \
    --dataset  "$DATASET_PATH" \
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
# STEP 5 — Batch turbulence inference
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[5/5] Batch inference"

OUTPUT_PATH=$(_cfg "output_path" "./outputs")
mkdir -p "$OUTPUT_PATH"

python3 run_batch.py --config "$CONFIG"

echo ""
echo "============================================================"
echo "  Done.  Outputs saved to: $OUTPUT_PATH"
echo "  $(date)"
echo "============================================================"
