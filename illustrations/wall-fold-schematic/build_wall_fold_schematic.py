"""Wall-and-fold schematic (poster Panel 2A): the wall at zero, the reflected
tail, and the tilted MSD slope it produces.

A pure matplotlib illustration -- no data input. The left panel shows a
fitted-variance distribution bounded at zero, with the forbidden negative tail
folded back across the wall so the distribution piles up and skews right (mean
above the true value). The right panel shows that per-frame bias growing as CNR
falls across late frames, tilting the MSD slope and overestimating D.

Each panel is written as its own image so the poster can place them
independently. Only the margin outside the axes is transparent -- the plot area
itself is opaque white -- so a panel drops onto the dark poster without a white
halo while its curves and labels stay readable. A combined two-panel image is
also written for callers that want both at once.

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

# Spanish flag palette (from the plot style guide). Three semantic
# colours: blue = the true value, red = the biased/inflated value, indigo = the
# fitted-variance distribution family (distribution of fits, unbounded estimate,
# reflected tail, forbidden tail), drawn at descending alpha so the one hue
# tells one story.
ROLE1 = "#0077c8"   # blue     -- the true value
ROLE4 = "#c60b1e"   # flag red -- the biased / inflated value
INDIGO = "#540d6e"  # indigo   -- the fitted-variance distribution family

HERE = Path(__file__).resolve().parent
OUT_LEFT = HERE / "wall_fold_schematic_left.png"
OUT_RIGHT = HERE / "wall_fold_schematic_right.png"
OUT_COMBINED = HERE / "wall_fold_schematic.png"


def _plot_left(axL):
    """Draw the fold-at-the-wall panel; return the curves the labels ride on.

    Plotting must precede ``tight_layout``, but the CurvedText labels need the
    settled display transform, so labelling is split into ``_label_left`` and
    called after layout.
    """
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

    axL.fill_between(v[pos], 0, g[pos], color=INDIGO, alpha=0.15, lw=0)
    axL.fill_between(v[pos], 0, g_ref[pos], color=INDIGO, alpha=0.15, lw=0)
    # One hue (the distribution family is one story); alpha ranks them: faint =
    # the hypothetical unbounded estimate, medium = the reflected mechanism,
    # full = the resulting distribution of fits.
    axL.plot(v[pos], g[pos], color=INDIGO, lw=1.8, alpha=0.40)             # realizable
    axL.plot(v[~pos], g[~pos], color=INDIGO, lw=1.4, alpha=0.40,
             ls=(0, (1, 1.6)))                                            # forbidden
    axL.plot(v[pos], g_ref[pos], color=INDIGO, lw=1.6, alpha=0.70)         # reflected
    axL.plot(v[pos], folded[pos], color=INDIGO, lw=1.8, ls=(0, (4, 2)))    # sum (fits)
    axL.axvline(0.0, color="k", lw=1.6)
    axL.axvline(m_true, color=ROLE1, ls=":", lw=1.1)
    axL.axvline(m_folded, color=ROLE4, ls=":", lw=1.1)

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
             fontsize=7, color=ROLE4, rotation=90, rotation_mode="anchor",
             va="center", ha="right")

    axL.set_xlabel("fitted variance,  $\\hat\\sigma^2$")
    axL.set_ylabel("probability density")
    axL.set_xlim(-3.0, 4.0)
    axL.set_ylim(0, 2.45)
    axL.set_xticks([])
    axL.set_yticks([])

    return {"v": v, "g": g, "g_ref": g_ref, "folded": folded}


def _label_left(axL, data):
    """Curve names run parallel to each curve and close to it. The rotation
    needs the settled display transform, so this is called after layout."""
    v, g, g_ref, folded = data["v"], data["g"], data["g_ref"], data["folded"]
    fm = (v >= -2.4) & (v <= -0.3)
    CurvedText(v[fm], g[fm], "forbidden:  variance < 0", axL, offset=5.0,
               pos=0.5, anchor="center", color=INDIGO, alpha=0.7, fontsize=7)
    sm = (v >= 0.3) & (v <= 2.3)
    CurvedText(v[sm], folded[sm], "distribution of fits", axL, offset=5.0,
               pos=0.6, anchor="center", color=INDIGO, fontsize=7)
    um = (v >= 1.0) & (v <= 3.2)
    CurvedText(v[um], g[um], "unbounded estimate", axL, offset=-6.0,
               pos=0.4, anchor="center", color=INDIGO, fontsize=7)
    rm = (v >= 0.4) & (v <= 2.4)
    CurvedText(v[rm], g_ref[rm], "reflected tail", axL, offset=5.5,
               pos=0.65, anchor="center", color=INDIGO, fontsize=7)


def _plot_right(axR):
    """Draw the tilted-MSD-slope panel; return the curves the labels ride on."""
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
    # layout (in _label_right), rotated to the line slope and pushed to opposite sides.
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

    return {"t": t, "true_line": true_line, "fit_line": fit_line}


def _label_right(axR, data):
    """Label both lines with CurvedText at the same fractional position on each
    line, offset to opposite sides so they hug their lines. Called after layout."""
    t, true_line, fit_line = data["t"], data["true_line"], data["fit_line"]
    CurvedText(t, fit_line, "inflated fit", axR, pos=0.4, anchor="center",
               offset=5.0, color=ROLE4, fontsize=7)
    CurvedText(t, true_line, "true slope $\\propto D$", axR, pos=0.4,
               anchor="center", offset=-5.0, color=ROLE1, fontsize=7)


def _save_opaque_axes(fig, path: Path) -> None:
    """Save with a transparent figure margin but an opaque white plot area.

    ``transparent=True`` blanks the axes patch too, leaving the curves floating
    on the dark poster. Instead we force each axes background to opaque white and
    save with a transparent figure facecolor, so only the region outside the plot
    proper lets the poster show through.
    """
    for ax in fig.axes:
        ax.set_facecolor("white")
    fig.savefig(path, dpi=300, facecolor="none")


def build_left() -> Path:
    """Standalone left panel (the fold at the wall); opaque plot, clear margin."""
    fig, axL = plt.subplots(figsize=(4.0, 3.4))
    data = _plot_left(axL)
    fig.tight_layout()
    _label_left(axL, data)
    _save_opaque_axes(fig, OUT_LEFT)
    plt.close(fig)
    return OUT_LEFT


def build_right() -> Path:
    """Standalone right panel (the tilted MSD slope); opaque plot, clear margin."""
    fig, axR = plt.subplots(figsize=(4.0, 3.4))
    data = _plot_right(axR)
    fig.tight_layout()
    _label_right(axR, data)
    _save_opaque_axes(fig, OUT_RIGHT)
    plt.close(fig)
    return OUT_RIGHT


def build_combined() -> Path:
    """Both panels side by side in one image; opaque plots, clear margin."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.0, 3.4))
    data_l = _plot_left(axL)
    data_r = _plot_right(axR)
    fig.tight_layout()
    _label_left(axL, data_l)
    _label_right(axR, data_r)
    _save_opaque_axes(fig, OUT_COMBINED)
    plt.close(fig)
    return OUT_COMBINED


def build() -> list[Path]:
    """Write both standalone panels and the combined image. Returns the paths."""
    return [build_left(), build_right(), build_combined()]


if __name__ == "__main__":
    for path in build():
        print(path)
