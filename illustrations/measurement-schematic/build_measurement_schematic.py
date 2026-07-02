"""Measurement schematic (poster Panel 1): a spreading, dimming population whose
per-frame contrast-to-noise falls frame by frame.

A pure matplotlib illustration -- no data input. The single physical curve is the
Gaussian profile from the simulation model (`weighting_wall.simulate.gaussian`);
everything else is annotation. Writes the figure into this folder.

Run:
    .venv/Scripts/python.exe illustrations/measurement-schematic/build_measurement_schematic.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from weighting_wall.simulate import gaussian  # noqa: E402

# Spanish-flag palette (see plot_style.md): the gradient runs blue (high CNR,
# early frame) to red (low CNR, late frame); the blend carries the time axis.
ROLE2 = "#0077c8"   # spanish-flag blue -- early frame (narrow, bright, high CNR)
ROLE4 = "#c60b1e"   # spanish-flag red  -- late frame  (wide, dim, low CNR)

OUT = Path(__file__).resolve().parent / "measurement_schematic.png"


def _lerp(c0: str, c1: str, f: float) -> tuple[float, float, float]:
    """Linear blend between two hex colours; f in [0, 1]."""
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple((1.0 - f) * a + f * b)


def build() -> Path:
    """Profiles broaden and dim across time; CNR falls frame by frame."""
    x = np.linspace(-6, 6, 400)
    times = np.linspace(0.0, 1.0, 5)
    sigmas = 1.0 + 2.2 * times          # broadening (diffusion)
    amps = np.exp(-1.3 * times)         # dimming (decay)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for f, sig, amp in zip(times, sigmas, amps):
        color = _lerp(ROLE2, ROLE4, f)
        ax.plot(x, gaussian(x, amp, 0.0, sig), color=color, lw=2.0)
    ax.annotate("", xy=(3.6, 0.92), xytext=(0.6, 0.92),
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(2.1, 0.96, "time", color="#444", fontsize=9, ha="center")
    ax.text(0.0, 1.02, "early frame:\nnarrow, bright (high CNR)",
            color=ROLE2, fontsize=8, ha="center", va="bottom")
    # Late-frame label sits in the centre-bottom gap, below the peaks, so it
    # clears the profile curves rather than overlapping them.
    ax.text(0.0, 0.08, "late frame:\nwide, dim (low CNR)",
            color=ROLE4, fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("position")
    ax.set_ylabel("intensity")
    ax.set_xlim(x[0], x[-1])             # curves run to the frame edges
    ax.set_ylim(0, 1.15)
    # A cartoon: axis labels only, no ticks or numbers.
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    # Transparent outside the plot only (plot_style.md): the figure margin
    # composites onto the poster background, while the axes patch stays opaque
    # white. Do not pass transparent=True -- that would clear the axes patch too.
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(build())
