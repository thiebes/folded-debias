"""Wall-and-fold schematic (poster Panel 2A): the wall at zero, the reflected
tail, and the tilted MSD slope it produces.

A pure matplotlib illustration -- no data input. Left panel: a fitted-variance
distribution bounded at zero, with the forbidden negative tail folded back across
the wall so the distribution piles up and skews right (mean above the true value).
Right panel: that per-frame bias grows as CNR falls across late frames, tilting the
MSD slope and overestimating D. Writes the figure into this folder.

Requires the sibling `curved-text` package (the curved on-curve labels):
    .venv/Scripts/python.exe -m pip install -e ../curved-text

Run:
    .venv/Scripts/python.exe illustrations/wall-fold-schematic/build_wall_fold_schematic.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from curved_text import CurvedText  # noqa: E402

# Poster palette (Okabe-Ito, colourblind-safe): blue = the true value,
# red = the biased/inflated value.
ROLE1 = "#0072B2"   # mediterranean blue -- the true value
ROLE4 = "#D55E00"   # spanish red        -- the biased / inflated value

OUT = Path(__file__).resolve().parent / "wall_fold_schematic.png"


def build() -> Path:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # --- Left: the fold at the wall ---------------------------------------
    v = np.linspace(-3.0, 4.0, 700)
    m_true, sd = 0.5, 1.1
    g_un = np.exp(-0.5 * ((v - m_true) / sd) ** 2)     # estimate with no wall
    g = g_un / g_un.max()
    pos = v >= 0
    # The forbidden negative tail, mirrored back across the wall (v -> -v).
    g_ref = np.exp(-0.5 * ((-v - m_true) / sd) ** 2) / g_un.max()
    folded = g + g_ref                                  # fitted-variance pdf

    # Reflection view: the forbidden tail (dotted, < 0) folds across the wall as
    # its own reflected curve. A light fill under both the realizable estimate and
    # the reflected tail overlaps near the wall, so the colour doubles there -- the
    # pile-up that skews the distribution. The distribution of fits is their sum
    # (dashed, in the fill colour).
    m_folded = (np.trapezoid(v[pos] * folded[pos], v[pos])
                / np.trapezoid(folded[pos], v[pos]))

    axL.fill_between(v[pos], 0, g[pos], color=ROLE4, alpha=0.15, lw=0)
    axL.fill_between(v[pos], 0, g_ref[pos], color=ROLE4, alpha=0.15, lw=0)
    # One hue (the three are one story); alpha ranks them: faint = the
    # hypothetical unbounded estimate, medium = the reflected mechanism, full =
    # the resulting distribution of fits.
    axL.plot(v[pos], g[pos], color=ROLE4, lw=1.8, alpha=0.40)              # realizable
    axL.plot(v[~pos], g[~pos], color=ROLE4, lw=1.4, alpha=0.40,
             ls=(0, (1, 1.6)))                                            # forbidden
    axL.plot(v[pos], g_ref[pos], color=ROLE4, lw=1.6, alpha=0.70)          # reflected
    axL.plot(v[pos], folded[pos], color=ROLE4, lw=1.8, ls=(0, (4, 2)))     # sum (fits)
    axL.axvline(0.0, color="k", lw=1.6)
    axL.axvline(m_true, color=ROLE1, ls=":", lw=1.1)
    axL.axvline(m_folded, color="#7a2f00", ls=":", lw=1.1)

    # The bias is the gap between the two; the names run vertically along their
    # own dotted lines. Bias sits to the right of the arrow, level with it.
    axL.text(-0.12, 1.25, "wall at 0", fontsize=8, rotation=90, va="center",
             ha="center", color="#333")
    axL.annotate("", xy=(m_folded, 1.66), xytext=(m_true, 1.66),
                 arrowprops=dict(arrowstyle="<->", color="k", lw=1.0,
                                 mutation_scale=8))
    axL.text(m_folded + 0.09, 1.66, "bias", fontsize=8, ha="left", va="center")
    axL.text(m_true - 0.13, 2.40, "true $\\sigma^2$", fontsize=7, color=ROLE1,
             rotation=90, rotation_mode="anchor", va="center", ha="right")
    axL.text(m_folded - 0.13, 2.40, "mean $\\langle\\hat\\sigma^2\\rangle$",
             fontsize=7, color="#7a2f00", rotation=90, rotation_mode="anchor",
             va="center", ha="right")

    axL.set_xlabel("fitted variance,  $\\hat\\sigma^2$")
    axL.set_ylabel("probability density")
    axL.set_xlim(-3.0, 4.0)
    axL.set_ylim(0, 2.45)
    axL.set_xticks([])
    axL.set_yticks([])

    # --- Right: the tilted MSD slope --------------------------------------
    t = np.linspace(0, 3.0, 7)
    slope_true = 0.04
    true_line = slope_true * t
    # Per-frame inflation grows as CNR falls (later frames more biased).
    inflation = 0.012 * (t / t.max()) ** 2 + 0.004
    biased = true_line + inflation
    fit = np.polyfit(t, biased, 1)
    fit_line = np.polyval(fit, t)

    axR.plot(t, true_line, color=ROLE1, lw=2.0)
    axR.plot(t, fit_line, color=ROLE4, lw=2.0)
    axR.plot(t, biased, "o", color=ROLE4, ms=5)
    for ti, tr, bi in zip(t, true_line, biased):
        axR.plot([ti, ti], [tr, bi], color="#bbb", lw=0.8, zorder=0)

    # The slope-gap annotation is the payload; the two line labels are added after
    # layout (below), rotated to the line slope and pushed to opposite sides.
    axR.annotate("", xy=(2.93, fit_line[-1]), xytext=(2.93, true_line[-1]),
                 arrowprops=dict(arrowstyle="<->", color="k", lw=0.9,
                                 mutation_scale=8))
    axR.text(2.84, (true_line[-1] + fit_line[-1]) / 2, "$D$ over-\nestimated",
             fontsize=7.5, ha="right", va="center")

    axR.set_xlabel("time")
    axR.set_ylabel("mean-squared displacement")
    axR.set_xlim(-0.08, 3.25)
    axR.set_ylim(-0.004, 0.155)
    axR.set_xticks([])
    axR.set_yticks([])

    fig.tight_layout()

    # Right panel: label both lines with CurvedText at the same fractional
    # position on each line, offset to opposite sides so they hug their lines.
    CurvedText(t, fit_line, "inflated fit", axR, pos=0.4, anchor="center",
               offset=5.0, color=ROLE4, fontsize=7)
    CurvedText(t, true_line, "true slope $\\propto D$", axR, pos=0.4,
               anchor="center", offset=-5.0, color=ROLE1, fontsize=7)

    # Curve names run parallel to each curve and close to it (the rotation needs
    # the settled display transform, so this comes after the layout).
    fm = (v >= -2.4) & (v <= -0.3)
    CurvedText(v[fm], g[fm], "forbidden:  variance < 0", axL, offset=5.0,
               pos=0.5, anchor="center", color=ROLE4, alpha=0.7, fontsize=7)
    sm = (v >= 0.3) & (v <= 2.3)
    CurvedText(v[sm], folded[sm], "distribution of fits", axL, offset=5.0,
               pos=0.6, anchor="center", color=ROLE4, fontsize=7)
    um = (v >= 1.0) & (v <= 3.2)
    CurvedText(v[um], g[um], "unbounded estimate", axL, offset=-6.0,
               pos=0.4, anchor="center", color=ROLE4, fontsize=7)
    rm = (v >= 0.4) & (v <= 2.4)
    CurvedText(v[rm], g_ref[rm], "reflected tail", axL, offset=5.5,
               pos=0.65, anchor="center", color=ROLE4, fontsize=7)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(build())
