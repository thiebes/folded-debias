"""Rise-then-decay drawn on a log x-axis -- a bare illustration.

A pure matplotlib illustration: a curve that rises briefly from zero and then
decays exponentially, over a single decade of t. The x-axis is logarithmic, the
y-axis linear. The y-axis is blank (no ticks, numbers, or label); the x-axis
carries only the label "t" with major and minor ticks spanning one decade and no
numbers. Both ranges are arbitrary and just encompass the line. No caption.

Run:
    .venv/Scripts/python.exe illustrations/rise-decay-semilog/build_rise_decay_semilog.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

# Spanish-flag palette (plot_style.md): red marks the low band, blue above it.
RED = "#c60b1e"   # Spanish flag red  -- bottom ~1/4 of the y-axis
BLUE = "#0077c8"  # Spanish-flag accent blue -- above the band

# Fraction of the y-axis (from the bottom) painted red.
LOW_BAND = 0.25

OUT = Path(__file__).resolve().parent / "rise_decay_semilog.png"

inch = 1 / 2.54


def _insert_crossings(t, y, level):
    """Insert exact (t, level) points wherever the curve crosses `level`.

    Linear interpolation along each segment that straddles `level`, so the
    returned polyline has a vertex sitting exactly on the threshold and no
    segment spans it -- the colour split is then clean at the boundary.
    """
    ts, ys = [t[0]], [y[0]]
    for i in range(len(t) - 1):
        y0, y1 = y[i], y[i + 1]
        if (y0 - level) * (y1 - level) < 0:          # strict sign change
            f = (level - y0) / (y1 - y0)
            ts.append(t[i] + f * (t[i + 1] - t[i]))
            ys.append(level)
        ts.append(t[i + 1])
        ys.append(y[i + 1])
    return np.array(ts), np.array(ys)


def build() -> Path:
    """One decade of t on a log x-axis; brief rise from zero then decay."""
    # One decade of t (1 -> 10) so the x-axis spans exactly one decade.
    t = np.linspace(1.0, 10.0, 400)
    # Rise-then-decay measured from the decade start: a fast rise (tau_rise)
    # from zero, then a slower exponential decay (tau_decay).
    u = t - t[0]
    y = (1.0 - np.exp(-u / 0.25)) * np.exp(-u / 3.5)

    ymax = y.max() * 1.05                 # y-axis top; line just fits below it
    threshold = LOW_BAND * ymax           # bottom band boundary in data units

    fig, ax = plt.subplots(layout="constrained", figsize=(7.0 * inch, 5.0 * inch))

    # Colour the line per segment: red where it sits in the bottom band of the
    # y-axis, blue above. Insert exact threshold crossings first so the colour
    # switches precisely at the dashed line (no over/undershoot on the steep
    # rise), then key each segment on its midpoint height.
    t, y = _insert_crossings(t, y, threshold)
    pts = np.column_stack([t, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_mid = 0.5 * (y[:-1] + y[1:])
    colors = np.where(seg_mid < threshold, RED, BLUE)
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=2.0))
    ax.autoscale_view()

    # Red dashed line marking the red/blue band boundary, full plot width.
    ax.axhline(threshold, color=RED, lw=1.3, ls="--")

    ax.set_xscale("log")

    # x-axis: span exactly the one decade the line occupies; label only.
    ax.set_xlim(1.0, 10.0)
    ax.set_xlabel("$t$", fontsize=11)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())

    # y-axis: linear, arbitrary range that just encompasses the line; fully blank.
    ax.set_ylim(0.0, ymax)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    ax.set_ylabel("")

    # Only minor ticks: the major ticks sit on the decade boundaries, which
    # coincide with the frame edges and read awkwardly, so suppress them.
    ax.tick_params(axis="both", which="major", length=0)
    ax.tick_params(axis="both", which="minor", direction="in",
                   top=False, right=False, length=6, width=1.3)

    # Transparent outside the axes; the plot interior itself stays white.
    ax.set_facecolor("white")
    fig.patch.set_alpha(0.0)

    fig.savefig(OUT, dpi=300, transparent=False, facecolor="none")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(build())
