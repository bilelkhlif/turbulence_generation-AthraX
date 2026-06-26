"""
select_diverse_clips.py — Annotation-driven diversity selection for VisDrone-VID clips.

Given the full VisDrone2019-VID valset (N sequences), selects a diverse 30%
subset that maximises coverage across the following axes:

  1. Object density    — objects per frame  (sparse ↔ crowded)
  2. Object scale      — mean bbox area relative to frame area  (close ↔ far / high altitude ↔ low)
  3. Category mix      — pedestrian-heavy vs. vehicle-heavy vs. mixed
  4. Temporal length   — prefer shorter clips to keep compute cost low
  5. Spatial spread    — bbox centroid spread across the frame  (proxy for scene complexity)

Selection algorithm
-------------------
  a) Parse each sequence's annotation txt and compute a feature vector.
  b) Normalise all features to [0, 1].
  c) Run greedy max-min distance selection (Gonzalez algorithm) in the
     normalised feature space to pick the most spread-out subset.
     This guarantees diversity without needing cluster labels.
  d) Among candidates of equal distance, prefer the shorter clip.

The result is written to a JSON file so run_batch.py can read it without
re-running the selection logic.

Usage
-----
    python select_diverse_clips.py \\
        --dataset  ./dataset/visdrone_vid_val \\
        --fraction 0.30 \\
        --out      ./dataset/selected_clips.json

    python select_diverse_clips.py --help
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── VisDrone-VID annotation column indices ────────────────────────────────
# format: frame_idx, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
#         score, object_category, truncation, occlusion
COL_FRAME    = 0
COL_LEFT     = 2
COL_TOP      = 3
COL_W        = 4
COL_H        = 5
COL_SCORE    = 6   # 1 = evaluated, 0 = ignored
COL_CATEGORY = 7

# VisDrone object categories
CAT_IGNORED     = 0
CAT_PEDESTRIAN  = 1
CAT_PEOPLE      = 2
CAT_BICYCLE     = 3
CAT_CAR         = 4
CAT_VAN         = 5
CAT_TRUCK       = 6
CAT_TRICYCLE    = 7
CAT_AWNING_TRI  = 8
CAT_BUS         = 9
CAT_MOTOR       = 10
CAT_OTHERS      = 11

VEHICLE_CATS  = {CAT_CAR, CAT_VAN, CAT_TRUCK, CAT_BUS, CAT_TRICYCLE, CAT_AWNING_TRI}
PERSON_CATS   = {CAT_PEDESTRIAN, CAT_PEOPLE}


# ─────────────────────────────────────────────────────────────────────────────
# Annotation parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_annotation(ann_path: Path) -> np.ndarray:
    """
    Returns (N, 10) float32 array of all annotation rows.
    Rows where score == 0 (ignored) are kept — we need them for density stats.
    """
    rows = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue
            try:
                rows.append([float(x) for x in parts[:10]])
            except ValueError:
                continue
    if not rows:
        return np.zeros((0, 10), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def _infer_frame_size(seq_dir: Path) -> Tuple[int, int]:
    """
    Return (width, height) of the first frame found in seq_dir.
    Falls back to (1920, 1080) if no images found (avoids cv2 import).
    """
    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in exts)
    if not frames:
        return 1920, 1080
    # Read only the header bytes with a minimal JPEG/PNG parser
    # to avoid pulling in cv2 at selection time.
    try:
        w, h = _image_size_fast(frames[0])
        return w, h
    except Exception:
        return 1920, 1080


def _image_size_fast(path: Path) -> Tuple[int, int]:
    """
    Read image dimensions without decoding the full image.
    Supports JPEG and PNG only (all VisDrone frames are JPEG).
    """
    with open(path, "rb") as f:
        header = f.read(24)

    if header[:8] == b"\x89PNG\r\n\x1a\n":
        # PNG: width/height at bytes 16-24
        w = int.from_bytes(header[16:20], "big")
        h = int.from_bytes(header[20:24], "big")
        return w, h

    if header[:2] == b"\xff\xd8":
        # JPEG: scan for SOF marker
        import struct
        with open(path, "rb") as f:
            f.read(2)  # SOI
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    break
                if marker[0] != 0xFF:
                    break
                seg_len = struct.unpack(">H", f.read(2))[0]
                if marker[1] in (0xC0, 0xC1, 0xC2):
                    f.read(1)  # precision
                    h, w = struct.unpack(">HH", f.read(4))
                    return w, h
                f.read(seg_len - 2)

    raise ValueError("Unsupported image format")


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(seq_name: str, ann_path: Path, seq_dir: Path) -> Dict:
    """
    Compute a feature vector for one sequence.

    Returns a dict with:
      - name          : sequence name
      - num_frames    : number of annotated frames
      - density       : mean objects per frame
      - scale         : mean (bbox_area / frame_area) for non-ignored boxes
      - person_ratio  : fraction of objects that are pedestrians/people
      - vehicle_ratio : fraction of objects that are vehicles
      - spread        : mean distance of bbox centroids from frame centre (norm)
      - features      : np.ndarray of [density, scale, person_ratio,
                         vehicle_ratio, spread, inv_length] used for selection
    """
    ann = _parse_annotation(ann_path)
    fw, fh = _infer_frame_size(seq_dir)
    frame_area = fw * fh

    if ann.shape[0] == 0:
        num_frames = 0
        density = vehicle_ratio = person_ratio = scale = spread = 0.0
    else:
        # Use only evaluated boxes (score == 1) for object-level stats
        valid = ann[ann[:, COL_SCORE] == 1]
        frame_ids = np.unique(ann[:, COL_FRAME]).astype(int)
        num_frames = len(frame_ids)

        if valid.shape[0] == 0:
            density = vehicle_ratio = person_ratio = scale = spread = 0.0
        else:
            density = valid.shape[0] / max(num_frames, 1)

            bbox_areas = valid[:, COL_W] * valid[:, COL_H]
            scale = float(np.mean(bbox_areas) / frame_area)

            cats = valid[:, COL_CATEGORY].astype(int)
            n_valid = len(cats)
            person_ratio  = float(np.sum(np.isin(cats, list(PERSON_CATS)))  / n_valid)
            vehicle_ratio = float(np.sum(np.isin(cats, list(VEHICLE_CATS))) / n_valid)

            cx = valid[:, COL_LEFT] + valid[:, COL_W] / 2.0  # centroid x
            cy = valid[:, COL_TOP]  + valid[:, COL_H] / 2.0  # centroid y
            # Normalised distance from frame centre
            dx = (cx - fw / 2.0) / (fw / 2.0)
            dy = (cy - fh / 2.0) / (fh / 2.0)
            spread = float(np.mean(np.sqrt(dx**2 + dy**2)))

    # inv_length: shorter clips score higher (prefer them in selection)
    inv_length = 1.0 / max(num_frames, 1)

    return {
        "name":          seq_name,
        "num_frames":    num_frames,
        "density":       density,
        "scale":         scale,
        "person_ratio":  person_ratio,
        "vehicle_ratio": vehicle_ratio,
        "spread":        spread,
        "inv_length":    inv_length,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Greedy max-min diversity selection (Gonzalez algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_diverse_subset(feature_matrix: np.ndarray, k: int) -> List[int]:
    """
    Select k indices from feature_matrix (n × d) so that the selected points
    are maximally spread in the feature space.

    Uses the Gonzalez greedy farthest-point algorithm:
      1. Seed with the index closest to the centroid.
      2. Iteratively add the point farthest from the current selected set.

    Returns a list of k integer indices into feature_matrix.
    """
    n = feature_matrix.shape[0]
    k = min(k, n)

    # Seed: point closest to the feature centroid
    centroid = feature_matrix.mean(axis=0)
    dists_to_centroid = np.linalg.norm(feature_matrix - centroid, axis=1)
    selected = [int(np.argmin(dists_to_centroid))]

    # min-dist from each point to the selected set
    min_dists = np.linalg.norm(feature_matrix - feature_matrix[selected[0]], axis=1)

    while len(selected) < k:
        farthest = int(np.argmax(min_dists))
        selected.append(farthest)
        new_dists = np.linalg.norm(feature_matrix - feature_matrix[farthest], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def select_clips(dataset_root: Path, fraction: float, out_path: Path) -> List[str]:
    seq_root = dataset_root / "sequences"
    ann_root = dataset_root / "annotations"

    if not seq_root.exists():
        raise FileNotFoundError(
            f"Sequences folder not found: {seq_root}\n"
            "Did you run download_visdrone.py first?"
        )

    sequences = sorted(p for p in seq_root.iterdir() if p.is_dir())
    if not sequences:
        raise RuntimeError(f"No sequence sub-folders found in {seq_root}")

    print(f"[select] Found {len(sequences)} sequences in {seq_root}")

    # ── Compute features for every sequence ──────────────────────────────────
    all_features = []
    for seq_dir in sequences:
        name = seq_dir.name
        ann_path = ann_root / f"{name}.txt"
        if not ann_path.exists():
            # Some zips store annotations without the prefix; try alternate names
            candidates = list(ann_root.glob(f"*{name}*"))
            ann_path = candidates[0] if candidates else ann_path

        feats = compute_features(name, ann_path, seq_dir)
        all_features.append(feats)
        print(f"  {name:30s}  frames={feats['num_frames']:5d}  "
              f"density={feats['density']:5.1f}  scale={feats['scale']:.4f}  "
              f"ped={feats['person_ratio']:.2f}  veh={feats['vehicle_ratio']:.2f}  "
              f"spread={feats['spread']:.3f}")

    # ── Build normalised feature matrix ──────────────────────────────────────
    # Columns: density, scale, person_ratio, vehicle_ratio, spread, inv_length
    raw = np.array([
        [f["density"], f["scale"], f["person_ratio"],
         f["vehicle_ratio"], f["spread"], f["inv_length"]]
        for f in all_features
    ], dtype=np.float64)

    # Normalise each column to [0, 1]; handle degenerate (constant) columns
    col_min = raw.min(axis=0)
    col_max = raw.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0  # avoid division by zero
    normed = (raw - col_min) / col_range

    # ── Select diverse subset ─────────────────────────────────────────────────
    k = max(1, math.ceil(len(sequences) * fraction))
    print(f"\n[select] Selecting {k}/{len(sequences)} clips "
          f"({fraction*100:.0f}%) using greedy max-min diversity…")

    selected_indices = greedy_diverse_subset(normed, k)
    selected_clips = [all_features[i]["name"] for i in selected_indices]

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n[select] Selected clips:")
    for name in sorted(selected_clips):
        f = next(x for x in all_features if x["name"] == name)
        print(f"  ✓ {name:30s}  frames={f['num_frames']:5d}  "
              f"density={f['density']:5.1f}  ped={f['person_ratio']:.2f}  "
              f"veh={f['vehicle_ratio']:.2f}")

    # ── Write selection manifest ──────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "total_sequences": len(sequences),
        "selected_count":  k,
        "fraction":        fraction,
        "selected_clips":  selected_clips,
        "feature_stats":   {
            f["name"]: {
                "num_frames":    f["num_frames"],
                "density":       round(f["density"],   3),
                "scale":         round(f["scale"],      5),
                "person_ratio":  round(f["person_ratio"], 3),
                "vehicle_ratio": round(f["vehicle_ratio"], 3),
                "spread":        round(f["spread"],     3),
            }
            for f in all_features
        },
    }
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n[select] Manifest written to {out_path}")

    return selected_clips


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select a diverse subset of VisDrone-VID clips using annotation metadata"
    )
    p.add_argument(
        "--dataset",
        default="./dataset/visdrone_vid_val",
        help="Root of the extracted VisDrone-VID valset (default: ./dataset/visdrone_vid_val)",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=0.30,
        help="Fraction of clips to keep (default: 0.30)",
    )
    p.add_argument(
        "--out",
        default="./dataset/selected_clips.json",
        help="Path to write the selection manifest JSON",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    select_clips(
        dataset_root=Path(args.dataset),
        fraction=args.fraction,
        out_path=Path(args.out),
    )
