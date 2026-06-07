"""Produce the data-driven poster figures and rough panel schematics.

Five figures in one place so the poster carries a single, consistent style
(DICE palette, ticks inside the frame):

  * Panel 1 schematic -- the measurement: a population that broadens (diffusion)
    and dims (decay) frame by frame, so the contrast-to-noise falls with time.
  * Panel 2A schematic -- the mechanism: a fitted variance cannot go negative,
    so the sampling distribution reflects at the wall at zero into a right-skewed
    tail; the inflated per-frame mean tilts the MSD slope upward.
  * Panel 4 -- the precision floor: sqrt(CRLB)/D vs CNR_0, computed live from the
    Fisher information.
  * Panel 5 -- the correction works: corrected mean(D_hat)/D_true vs CNR_0 for the
    folded-normal scheme across the three profile shapes, parsed from the decile
    report in ``output/dense_calibration_and_test.txt``.
  * Panel 5 inset (Fig F) -- where it works: per-profile residual vs CNR for the
    three shapes, parsed from ``output/per_profile_floor_*.txt``.

The schematics are rough starting points meant to be redrawn by hand. The figures
that already have a rendered .png (boundary diagnostic, OLS/inverse-variance bias
curves) are not produced here.

Run from the repository root::

    .venv/Scripts/python.exe scripts/poster_figures.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from weighting_wall.simulate import Experiment, gaussian  # noqa: E402
from precision_limit.crlb import crlb_D  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)

# --- DICE palette (Montana State; from the sibling DICE repo) ----------------
DICE_BLUE = "#003f7f"
DICE_GOLD = "#f7941e"
DICE_GREEN = "#0cce6b"

FWHM0 = 2.3548                       # FWHM / sigma for a Gaussian
SHAPE_LABEL = {
    "gaussian": "Gaussian",
    "voigt": "Voigt",
    "generalized_gaussian": "generalized-Gaussian",
    "genGauss": "generalized-Gaussian",
}
# One DICE colour per profile shape: blue / gold / green.
SHAPE_COLOR = {
    "gaussian": DICE_BLUE,
    "voigt": DICE_GOLD,
    "generalized_gaussian": DICE_GREEN,
    "genGauss": DICE_GREEN,
}


def _style(ax) -> None:
    """Apply the shared axis style: ticks inside the frame, on all four sides."""
    ax.tick_params(direction="in", which="both", top=True, right=True)


def _lerp(c0: str, c1: str, f: float) -> tuple[float, float, float]:
    """Linear blend between two hex colours; f in [0, 1]."""
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple((1.0 - f) * a + f * b)


# --------------------------------------------------------------------------
# Panel 1 schematic -- spreading, dimming population
# --------------------------------------------------------------------------
def figure_schematic_measurement() -> Path:
    """Profiles broaden and dim across time; CNR falls frame by frame."""
    x = np.linspace(-6, 6, 400)
    times = np.linspace(0.0, 1.0, 5)
    sigmas = 1.0 + 2.2 * times          # broadening (diffusion)
    amps = np.exp(-1.3 * times)         # dimming (decay)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for f, sig, amp in zip(times, sigmas, amps):
        color = _lerp(DICE_BLUE, DICE_GOLD, f)
        ax.plot(x, gaussian(x, amp, 0.0, sig), color=color, lw=2.0)
    ax.annotate("", xy=(3.6, 0.92), xytext=(0.6, 0.92),
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(2.1, 0.96, "time", color="#444", fontsize=9, ha="center")
    ax.text(0.0, 1.02, "early frame:\nnarrow, bright (high CNR)",
            color=DICE_BLUE, fontsize=8, ha="center", va="bottom")
    ax.text(4.6, 0.30, "late frame:\nwide, dim (low CNR)",
            color=DICE_GOLD, fontsize=8, ha="center")
    ax.set_xlabel("position")
    ax.set_ylabel("intensity")
    ax.set_title("Diffusion broadens and decay dims each frame", fontsize=10)
    ax.set_ylim(0, 1.15)
    _style(ax)
    fig.tight_layout()
    path = OUTPUT / "poster_panel1_measurement.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Panel 2A schematic -- the wall at zero, the fold, the tilted slope
# --------------------------------------------------------------------------
def figure_schematic_wall_fold() -> Path:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # --- Left: the fold at the wall ---------------------------------------
    v = np.linspace(-3.0, 5.0, 600)
    m_true, sd = 0.5, 1.1
    g = np.exp(-0.5 * ((v - m_true) / sd) ** 2)        # unconstrained noise
    g /= g.max()
    # Folded (reflected at zero): mass that would fall below 0 reflects up.
    pos = v >= 0
    neg = v < 0
    folded = (np.exp(-0.5 * ((v - m_true) / sd) ** 2)
              + np.exp(-0.5 * ((-v - m_true) / sd) ** 2))
    folded /= g.max()

    # The forbidden negative tail, and the resulting folded distribution.
    axL.fill_between(v[neg], 0, g[neg], color=DICE_GOLD, alpha=0.35,
                     label="mass that would be < 0")
    axL.plot(v, g, color="#999", ls="--", lw=1.5, label="unconstrained noise")
    axL.plot(v[pos], folded[pos], color=DICE_BLUE, lw=2.4,
             label="fitted variance (folded)")
    axL.axvline(0.0, color="k", lw=1.6)
    axL.annotate("reflects\nto +", xy=(0.45, 0.62), xytext=(-1.7, 0.78),
                 fontsize=8, color="#b5710a", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#b5710a",
                                 connectionstyle="arc3,rad=-0.4"))
    axL.text(-0.06, 1.18, "wall at 0", fontsize=8, rotation=90, va="top",
             ha="right")

    # True value and inflated (folded) mean, labelled at the top inside.
    m_folded = np.trapezoid(v[pos] * folded[pos], v[pos]) / np.trapezoid(
        folded[pos], v[pos])
    axL.axvline(m_true, color=DICE_GREEN, ls=":", lw=1.8)
    axL.axvline(m_folded, color=DICE_GOLD, ls=":", lw=1.8)
    axL.annotate("true", xy=(m_true, 1.02), color=DICE_GREEN, fontsize=8,
                 ha="center", va="bottom")
    axL.annotate("biased mean", xy=(m_folded, 1.02), color="#b5710a",
                 fontsize=8, ha="center", va="bottom")
    axL.set_xlabel(r"estimated variance  $\hat\sigma^2$")
    axL.set_ylabel("density")
    axL.set_title("Reflection at the wall", fontsize=10)
    axL.set_xlim(-3.0, 5.0)
    axL.set_ylim(0, 1.32)
    axL.legend(fontsize=7, loc="upper right")
    _style(axL)

    # --- Right: the tilted MSD slope --------------------------------------
    t = np.linspace(0, 3.0, 7)
    slope_true = 0.04
    true_line = slope_true * t
    # Per-frame inflation grows as CNR falls (later frames more biased).
    inflation = 0.012 * (t / t.max()) ** 2 + 0.004
    biased = true_line + inflation
    fit = np.polyfit(t, biased, 1)

    axR.plot(t, true_line, color=DICE_BLUE, lw=2.0, label="true slope $\\propto D$")
    axR.plot(t, np.polyval(fit, t), color=DICE_GOLD, lw=2.0,
             label="fitted slope (inflated)")
    axR.plot(t, biased, "o", color=DICE_GOLD, ms=5)
    for ti, tr, bi in zip(t, true_line, biased):
        axR.plot([ti, ti], [tr, bi], color="#bbb", lw=0.8, zorder=0)
    axR.set_xlabel("time")
    axR.set_ylabel("MSD  (width$^2 - $width$^2_0$)")
    axR.set_title("Per-frame bias tilts the slope", fontsize=10)
    axR.legend(fontsize=7, loc="upper left")
    _style(axR)

    fig.tight_layout()
    path = OUTPUT / "poster_panel2_wall_fold.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Panel 4 -- precision floor
# --------------------------------------------------------------------------
def figure_crlb() -> Path:
    """sqrt(CRLB)/D vs CNR_0 at the poster's working D (L_D/FWHM_0 ~ 0.085)."""
    exp = Experiment()                 # defaults: D = 0.02 sigma_0^2/tau
    ld_over_fwhm = np.sqrt(2.0 * exp.D) / FWHM0
    cnr0 = np.logspace(np.log10(2.0), np.log10(100.0), 60)
    rel = np.array([np.sqrt(crlb_D(exp, c)) / exp.D for c in cnr0])

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.loglog(cnr0, rel, color=DICE_BLUE, lw=2.2)
    ax.axhline(1.0, color="#999999", ls="--", lw=1.0)
    ax.fill_between(cnr0, 1.0, rel, where=(rel >= 1.0),
                    color=DICE_GOLD, alpha=0.20)
    ax.annotate("a single measurement\ncannot pin D (SD > D)",
                xy=(3.0, 3.0), fontsize=8, color="#b5710a")
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$")
    ax.set_ylabel(r"$\sqrt{\mathrm{CRLB}}\,/\,D$  (best possible SD, units of $D$)")
    ax.set_title(f"Precision floor  ($L_D/\\mathrm{{FWHM}}_0 \\approx {ld_over_fwhm:.3f}$)",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.20)
    _style(ax)
    fig.tight_layout()
    path = OUTPUT / "poster_panel4_crlb.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Panel 5 -- corrected bias curves (folded-normal), three shapes
# --------------------------------------------------------------------------
_DECILE_RE = re.compile(
    r"\[\s*([\d.]+),\s*([\d.]+)\]\s+(\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def _parse_dense_deciles(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Parse the per-shape decile report (mean(D_hat)/D_true per scheme).

    Returns ``{shape: {"cnr": array, "folded": array}}`` using the geometric
    midpoint of each CNR_0 decile as the x value.
    """
    text = path.read_text()
    shapes: dict[str, dict[str, np.ndarray]] = {}
    current = None
    cnr, folded = [], []
    for line in text.splitlines():
        header = re.match(r"=== (\w+)\s", line)
        if header:
            if current is not None and cnr:
                shapes[current] = {"cnr": np.array(cnr),
                                   "folded": np.array(folded)}
            current = header.group(1)
            cnr, folded = [], []
            continue
        m = _DECILE_RE.search(line)
        if m and current is not None:
            lo, hi = float(m.group(1)), float(m.group(2))
            cnr.append(np.sqrt(lo * hi))            # geometric midpoint
            folded.append(float(m.group(6)))         # folded column
    if current is not None and cnr:
        shapes[current] = {"cnr": np.array(cnr), "folded": np.array(folded)}
    return shapes


def figure_corrected_bias(dense_path: Path) -> Path | None:
    if not dense_path.exists():
        print(f"  ({dense_path.name} not found -- skipping Panel 5 curve)")
        return None
    shapes = _parse_dense_deciles(dense_path)
    if not shapes:
        print(f"  (no decile data parsed from {dense_path.name} -- skipping)")
        return None

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axhspan(0.95, 1.05, color="#cccccc", alpha=0.35,
               label=r"$\pm 0.05$ band")
    ax.axhline(1.0, color="#666666", lw=1.0)
    for shape, d in shapes.items():
        ax.semilogx(d["cnr"], d["folded"], "o-", ms=4, lw=1.8,
                    color=SHAPE_COLOR.get(shape, DICE_BLUE),
                    label=SHAPE_LABEL.get(shape, shape))
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$")
    ax.set_ylabel(r"corrected  $\langle \hat D\rangle / D_\mathrm{true}$")
    ax.set_title("Folded-normal correction across profile shapes", fontsize=10)
    ax.set_ylim(0.7, 1.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.20)
    _style(ax)
    fig.tight_layout()
    path = OUTPUT / "poster_panel5_corrected_bias.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Panel 5 inset (Fig F) -- per-profile floor, three shapes
# --------------------------------------------------------------------------
def _parse_floor(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse (CNR, residual, SE) from a per-profile-floor output file.

    A data row is ``CNR n raw_mean corrected_mean residual SE |z| [flag]``;
    we take columns CNR (0), residual (4), SE (5).
    """
    cnr, resid, se = [], [], []
    for line in path.read_text().splitlines():
        tok = line.split()
        if len(tok) >= 7 and tok[1] == "50000":
            try:
                cnr.append(float(tok[0]))
                resid.append(float(tok[4]))
                se.append(float(tok[5]))
            except ValueError:
                continue
    return np.array(cnr), np.array(resid), np.array(se)


def figure_floor() -> Path | None:
    files = {
        "gaussian": OUTPUT / "per_profile_floor_gaussian.txt",
        "voigt": OUTPUT / "per_profile_floor_voigt.txt",
        "genGauss": OUTPUT / "per_profile_floor_genGauss.txt",
    }
    present = {k: v for k, v in files.items() if v.exists()}
    if not present:
        print("  (no per_profile_floor_*.txt found -- skipping Fig F)")
        return None

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    # Regime dividers (per-profile CNR, from STATUS / DERIVATION sec. 9), kept
    # neutral so the DICE-coloured shape curves read cleanly.
    for xline in (0.7, 1.0):
        ax.axvline(xline, color="#bbbbbb", ls="--", lw=1.0)
    ax.text(0.42, 1.25, "failing", color="#777", fontsize=8, ha="center")
    ax.text(0.83, 1.25, "useful", color="#777", fontsize=8, ha="center")
    ax.text(6.0, 1.25, "reliable", color="#777", fontsize=8, ha="center")

    for shape, path in present.items():
        cnr, resid, se = _parse_floor(path)
        order = np.argsort(cnr)
        ax.semilogx(cnr[order], np.abs(resid[order]), "o-", ms=3, lw=1.6,
                    color=SHAPE_COLOR.get(shape, DICE_BLUE),
                    label=SHAPE_LABEL.get(shape, shape))
    ax.set_xlabel(r"per-profile contrast-to-noise  CNR")
    ax.set_ylabel(r"$|$residual$|$  (in $\sigma_0^2$)")
    ax.set_title("Where the correction works", fontsize=10)
    ax.set_xlim(0.2, 100)
    ax.set_ylim(0, 1.4)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.20)
    _style(ax)
    fig.tight_layout()
    path = OUTPUT / "poster_panel5_floor.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    print("Producing poster figures...")
    print(f"  Panel 1 schematic (measurement):   {figure_schematic_measurement().name}")
    print(f"  Panel 2A schematic (wall + fold):  {figure_schematic_wall_fold().name}")
    print(f"  Panel 4 (precision floor):         {figure_crlb().name}")
    p5 = figure_corrected_bias(OUTPUT / "dense_calibration_and_test.txt")
    if p5:
        print(f"  Panel 5 (corrected bias):          {p5.name}")
    pf = figure_floor()
    if pf:
        print(f"  Panel 5 inset (per-profile floor): {pf.name}")
    print("Done.")


if __name__ == "__main__":
    main()
