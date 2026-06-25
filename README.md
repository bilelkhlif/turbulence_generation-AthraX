# Turbulence Generation — AthraX

A tool for adding physically accurate atmospheric turbulence to video, built on top of the P2S (Phase-to-Space) simulator from Mao et al., ICCV 2021. It includes a Streamlit interface for running and benchmarking different turbulence configurations without touching the command line.

---

## Background

Atmospheric turbulence degrades images captured through long optical paths — surveillance cameras, drones observing ground targets, and long-range tracking systems all suffer from it. The degradation has two components: spatially varying blur caused by random wavefront aberrations (higher-order effects), and pixel displacement caused by random tilt of the wavefront (tip-tilt effect).

The P2S method replaces slow wave-optics simulation with a pre-trained neural network that maps Zernike coefficients directly to PSF weights, making it fast enough to process full videos.

**Reference:** Z. Mao, N. Chimitt, S. H. Chan, *Accelerating Atmospheric Turbulence Simulation via Learned Phase-to-Space Transform*, ICCV 2021. [[arXiv]](https://arxiv.org/abs/2107.11627)

---

## Repository Structure

```
.
├── app.py                        # Streamlit interface
├── apply_turbulence_to_video.py  # Video processing pipeline (CLI)
├── simulator.py                  # P2S simulator (modified: CPU support)
├── turbStats.py                  # Tilt and PSF correlation matrix generation
├── .streamlit/
│   └── config.toml               # Streamlit config (disables telemetry)
└── data/
    ├── dictionary.npy            # PSF basis kernels (pre-trained, required)
    ├── P2S_model.pt              # Neural network weights (required)
    ├── R-corr_*.npy              # PSF spatial correlation matrices (auto-generated)
    ├── S_half-*.npy              # Tilt correlation matrices (auto-generated)
    └── *.mp4                     # Input and output videos (not tracked)
```

The `.npy` matrix files and `.mp4` videos are excluded from version control (see `.gitignore`). The two required model files (`dictionary.npy`, `P2S_model.pt`) must be downloaded separately — see setup below.

---

## Setup

**Requirements:** Python 3.9+

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python scipy streamlit
```

If you have a CUDA-capable GPU, install the CUDA build of PyTorch instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Download the required model files** and place them in `data/`:

- `data/dictionary.npy` — PSF basis dictionary
- `data/P2S_model.pt` — P2S network weights

These are available from the [original repository](https://github.com/Riponcs/TurbulenceSimulatorPython) or the authors' [project page](https://engineering.purdue.edu/ChanGroup/project_turbulence.html).

---

## Usage

### Option 1 — Streamlit interface

Place your input video in the `data/` folder, then launch:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser. From there you can:

- Select an input video and name the output
- Adjust all turbulence parameters using sliders
- Run the simulation and watch the console output in real time
- Preview the output video directly in the browser
- See a table of all previous outputs for comparison

### Option 2 — Command line

```bash
python apply_turbulence_to_video.py \
  --input  data/your_video.mp4 \
  --output data/output.mp4 \
  --D 0.1 --r0 0.2 \
  --scale 1.0 \
  --corr -0.1 \
  --img_size 256 \
  --L 3000
```

---

## Parameters

### D / r0 — master strength dial

The ratio of aperture diameter (D) to the Fried coherence length (r0). This is the primary control for how strong the turbulence appears. A larger ratio means more turbulence cells across the aperture, which produces stronger blur and larger pixel displacements.

In the simulator, all random Zernike coefficients are scaled by `(D/r0)^(5/3)`, following Kolmogorov turbulence statistics. This affects both the blur and the displacement simultaneously.

| D/r0 | Visual severity |
|------|----------------|
| 0.1 – 0.4 | barely visible, heat shimmer |
| 0.5 – 1.0 | subtle, realistic long-range camera |
| 1.0 – 2.0 | moderate, visible distortion |
| 2.0 – 3.5 | strong, desert highway mirage |
| 3.5 – 5.0 | severe |

The first time you use a new D/r0 value, the tilt matrix (`S_half-*.npy`) is computed and cached in `data/`. This takes a few minutes but only happens once per unique value.

### Scale

A multiplier applied to the Zernike coefficient vector on top of the D/r0 scaling. Use it for fine-grained control without needing to generate a new tilt matrix. Values below 1.0 reduce the effect; values above 1.0 amplify it.

### PSF spatial correlation (corr)

Controls how similar the blur kernel is between neighbouring regions of the frame. Mathematically, it is the decay rate of the spatial covariance of the Zernike coefficients across the 16×16 sub-aperture grid.

| Value | Effect |
|-------|--------|
| -0.01 | strongly correlated — uniform-looking blur across the frame |
| -0.1  | moderate spatial variation (default) |
| -1.0  | noticeable patch-to-patch differences |
| -5.0  | maximum spatial variation — most "patchy" appearance |

Switching to a new corr value for the first time triggers generation of `R-corr_*.npy`, which can take up to 10 minutes. The `-0.1` matrix is generated on first run and is suitable for most use cases.

### Internal simulation size (img_size)

The input frame is resized to this square resolution before the simulator processes it, then resized back to the original dimensions for output. Larger values produce more spatially detailed blur and displacement but increase processing time quadratically.

| Size | Recommendation |
|------|---------------|
| 128 | quick parameter sweeps |
| 256 | default — good balance on CPU |
| 512 | high quality, slow on CPU |
| 1024 | very high quality, requires GPU |

### Propagation distance L (metres)

Used only during tilt matrix precomputation. It sets the spatial scale of the displacement field — longer distances produce larger-scale wavefront structure. Typical values are 500–2000 m for urban surveillance and 2000–10000 m for long-range scenarios.

Once the tilt matrix is cached for a given (img_size, D/r0, L) combination, changing L has no effect until a new matrix is generated with the new value.

---

## First-Run Behaviour

The first time you run with a given set of parameters, the script will generate and cache:

1. **PSF correlation matrix** (`R-corr_{corr}.npy`) — computed once per `corr` value, takes ~10 minutes
2. **Tilt matrix** (`S_half-size_{N}-D_r0_{Dr0}.npy`) — computed once per (img_size, D/r0, L) combination, takes a few minutes

Subsequent runs with the same parameters skip both steps and go straight to frame processing.

---

## Notes

- The simulator was originally written for GPU. This fork adds `map_location='cpu'` to support CPU-only machines.
- Output videos are encoded with the `mp4v` codec. If playback fails in some players, re-encode with `ffmpeg -i output.mp4 -vcodec libx264 output_h264.mp4`.
- The tool processes one frame at a time and does not support batching.
