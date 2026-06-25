"""
Atmospheric Turbulence Simulator — Streamlit UI
Run:  python -m streamlit run app.py
"""

import os, sys, time, subprocess
from pathlib import Path

import streamlit as st

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Turbulence Simulator", layout="wide")

st.title("Atmospheric Turbulence Simulator")
st.caption("P2S · Mao et al., ICCV 2021 — adds physically accurate turbulence to video")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_run, tab_docs = st.tabs(["Run", "Parameter Reference"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ══════════════════════════════════════════════════════════════════════════════
with tab_run:

    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: controls ────────────────────────────────────────────────────────
    with left:

        # Input / output
        st.subheader("Files")
        mp4_files = sorted(
            f.name for f in DATA_DIR.glob("*.mp4")
            if not f.name.startswith("output_")
        )
        if not mp4_files:
            st.error("No input videos found in data/. Add an .mp4 file there first.")
            st.stop()

        input_video = st.selectbox("Input video", mp4_files)
        output_name = st.text_input("Output filename", value="output_turbulence.mp4")
        if not output_name.endswith(".mp4"):
            output_name += ".mp4"

        st.divider()

        # ── Turbulence strength ───────────────────────────────────────────────
        st.subheader("Turbulence Strength")

        Dr0 = st.slider(
            "D / r0",
            min_value=0.1, max_value=5.0, value=0.5, step=0.1,
            help="Master strength dial. Controls both blur and pixel displacement. "
                 "Below 1.0 is very subtle; 2–3 is moderate; 5 is severe."
        )
        st.caption(
            f"D/r0 = **{Dr0:.1f}** — "
            + {True: "very subtle", False: ""}[Dr0 < 0.5]
            + {True: "subtle", False: ""}[0.5 <= Dr0 < 1.0]
            + {True: "mild", False: ""}[1.0 <= Dr0 < 1.5]
            + {True: "moderate", False: ""}[1.5 <= Dr0 < 2.5]
            + {True: "strong", False: ""}[2.5 <= Dr0 < 4.0]
            + {True: "severe", False: ""}[Dr0 >= 4.0]
        )

        scale = st.slider(
            "Scale",
            min_value=0.1, max_value=3.0, value=1.0, step=0.1,
            help="Multiplier applied on top of D/r0. "
                 "Values below 1.0 reduce the effect further without changing the physical model."
        )

        st.divider()

        # ── PSF correlation ───────────────────────────────────────────────────
        st.subheader("Blur Texture")

        corr_options = {
            "-0.01  (strongly correlated — uniform blur)": -0.01,
            "-0.1   (moderate correlation) [default]":     -0.1,
            "-1.0   (weak correlation — patchy blur)":      -1.0,
            "-5.0   (independent patches — most spatially varied)": -5.0,
        }
        corr_label = st.selectbox(
            "PSF spatial correlation",
            list(corr_options.keys()),
            index=1,
            help="Controls how similar the blur kernel is across different parts of the frame."
        )
        corr = corr_options[corr_label]

        st.divider()

        # ── Internal resolution ───────────────────────────────────────────────
        st.subheader("Quality / Speed")

        img_size = st.select_slider(
            "Internal simulation size",
            options=[128, 256, 512, 1024],
            value=256,
            help="The frame is resized to this square before simulation. "
                 "Higher = better quality, much slower on CPU."
        )

        L_val = st.number_input(
            "Propagation distance L (metres)",
            min_value=100, max_value=20000, value=3000, step=100,
            help="Used only when generating the tilt matrix for a new D/r0 value. "
                 "Represents how far light travels through the atmosphere."
        )

        st.divider()

        run = st.button("Run", type="primary", use_container_width=True)

    # ── RIGHT: output ─────────────────────────────────────────────────────────
    with right:

        st.subheader("Console")
        log_area = st.empty()

        st.subheader("Result")
        video_area = st.empty()

        if run:
            D_val  = 0.1
            r0_val = round(D_val / Dr0, 6)

            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "apply_turbulence_to_video.py"),
                "--input",    str(DATA_DIR / input_video),
                "--output",   str(DATA_DIR / output_name),
                "--D",        str(D_val),
                "--r0",       str(r0_val),
                "--L",        str(L_val),
                "--img_size", str(img_size),
                "--corr",     str(corr),
                "--scale",    str(scale),
            ]

            log_lines = []
            with st.spinner("Processing..."):
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(SCRIPT_DIR),
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        log_lines.append(line)
                        log_area.code("\n".join(log_lines[-30:]))
                proc.wait()

            if proc.returncode == 0:
                out_path = DATA_DIR / output_name
                if out_path.exists():
                    video_area.video(str(out_path))
                    st.success(f"Saved to data/{output_name}")
            else:
                st.error("Simulation failed — see console output above.")

        # previous outputs table
        outputs = sorted(
            (f for f in DATA_DIR.glob("*.mp4") if f.name != input_video),
            key=lambda f: f.stat().st_mtime, reverse=True
        )
        if outputs:
            st.divider()
            st.subheader("Previous outputs")
            rows = [
                {
                    "File": f.name,
                    "Size (MB)": f"{f.stat().st_size / 1_048_576:.1f}",
                    "Date": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
                }
                for f in outputs
            ]
            st.table(rows)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PARAMETER REFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:

    st.subheader("Parameter Reference")
    st.markdown("""
These are **all** the parameters that affect the simulation output. Nothing has been omitted
and nothing irrelevant is exposed.

---

### D / r0 — Turbulence strength (master dial)

**What it is.**
The ratio of the aperture diameter D to the Fried coherence length r0.
r0 is a measure of atmospheric coherence — large r0 means the atmosphere is calm,
small r0 means it is turbulent. D/r0 therefore captures how many "turbulence cells"
fit across the aperture.

**What it controls in the code.**
In `simulator.py → forward()`, the random Zernike coefficients are scaled by
`Dr0^(5/3)` (the 5/3 power comes directly from Kolmogorov turbulence theory).
This scaling drives **both** the blur (via the P2S network) and the pixel
displacement (via the tilt field). There is no way to change one without the other
through D/r0 alone — that is physically correct.

**Practical range.**
- 0.1 – 0.4 : barely visible, heat-shimmer level  
- 0.5 – 1.0 : subtle, realistic long-range camera  
- 1.0 – 2.0 : moderate, visible distortion  
- 2.0 – 3.5 : strong, desert highway mirage  
- 3.5 – 5.0 : severe, near-unusable imaging

The tilt matrix (`S_half-*.npy`) is precomputed per unique D/r0 value and cached in
`data/`. The first run for a new value takes a few minutes; subsequent runs are
instant.

---

### Scale — tilt multiplier

**What it is.**
A scalar that multiplies the Zernike coefficient vector *after* the D/r0 scaling
but *before* the P2S network maps them to PSF weights.

**What it controls in the code.**
In `forward()`: `zer = zer * self.scale`. This amplifies or attenuates the
Zernike modes, which feed both the blur network and contribute to the displacement
field. It is an extra knob on top of D/r0 — useful when you want finer control
without generating a new tilt matrix.

**Practical range.**
- 0.1 – 0.5 : reduces the effect, good for getting very subtle turbulence at low D/r0  
- 1.0 : neutral (no change)  
- 1.5 – 3.0 : amplifies beyond what D/r0 alone produces

---

### PSF spatial correlation (corr)

**What it is.**
Controls how similar the blur kernel (Point Spread Function) is between
neighbouring patches of the frame.

**What it controls in the code.**
The matrix R is loaded from `R-corr_{corr}.npy`. It is the square root of the
spatial covariance matrix of the Zernike coefficients across the 16×16 sub-aperture
grid. When |corr| is small (e.g. -0.01) the covariance is high — neighbouring
patches get nearly identical blur kernels, producing a smooth, uniform-looking
degradation. When |corr| is large (e.g. -5) the covariance falls off quickly —
each patch gets an independent blur, producing a spatially varied, "patchy" look.

**Practical values.**
- -0.01 : uniform blur, as if seen through a single large lens  
- -0.1  : mild spatial variation (default — physically representative)  
- -1.0  : noticeable patch-to-patch variation  
- -5.0  : maximum spatial variation, most "broken" looking

Note: this matrix is also cached per corr value. If you switch to -1.0 for the first
time it will take ~10 minutes to generate. -0.1 was already generated on your first run.

---

### Internal simulation size (img\_size)

**What it is.**
The resolution of the square image the simulator operates on. The input frame is
resized to this square before simulation, then resized back to the original
dimensions for output.

**What it controls.**
A larger size means more spatial detail in the blur and displacement fields, but
the computation cost scales with the square of the size (256 is 4× faster than 512).
This is purely a quality/speed tradeoff — it does not change the physical model.

**Practical values.**
- 128 : fast, low quality — good for quick parameter sweeps  
- 256 : good balance for most videos (recommended on CPU)  
- 512 : high quality, noticeably slower on CPU  
- 1024 : very high quality, impractical without a GPU

---

### Propagation distance L (metres)

**What it is.**
The distance light travels through the turbulent medium (e.g. camera-to-subject range).

**What it controls.**
L is used only during tilt matrix precomputation (`tilt_mat()` in `turbStats.py`).
It sets the isoplanatic scale of the displacement field — larger L means the
displacement pattern has larger spatial structure. Once the tilt matrix is cached for
a given (img\_size, D/r0, L) combination, changing L has no effect until a new matrix
is generated.

**Practical range.**
- 500 – 1000 m : short range, e.g. traffic camera  
- 1000 – 5000 m : medium range, typical surveillance  
- 5000+ m : long range, military or astronomical imaging
""")
