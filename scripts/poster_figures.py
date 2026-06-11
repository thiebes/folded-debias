"""Produce the data-driven poster figures and rough panel schematics.

Five figures in one place so the poster carries a single, consistent style
(cubehelix poster palette, ticks inside the frame):

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

# The fft-cnr validation lives in the sibling repo (see POSTER_PLAN); the Panel 4
# CNR-recovery figure is restyled from its case-11 sweep. Prefer the dense
# poster-local sweep (scripts/poster_fftcnr_dense.py) when present, so the ratio
# curve is smooth; otherwise fall back to the sibling's canonical six-point file.
_FFTCNR_DENSE = OUTPUT / "case11_gengauss_cnr_sweep_dense.npz"
_FFTCNR_CASE11 = (Path(__file__).resolve().parents[2] / "fft-cnr-paper"
                  / "validation" / "results" / "case11_gengauss_cnr_sweep.npz")
FFTCNR_NPZ = _FFTCNR_DENSE if _FFTCNR_DENSE.exists() else _FFTCNR_CASE11

# --- Poster palette (cubehelix role colours) ---------------------------------
# A cubehelix-derived sequence chosen for the poster: five roles that climb in
# lightness monotonically, so the ordering survives colour-vision deficiency and
# grayscale printing. Roles 1-3 (dark green, blue, purple) read crisply on white
# and carry the primary data lines; role 4 (rose) is the contrast partner for the
# true value; role 5 (sand, lightest) is reserved for fills and region markers,
# not thin standalone lines. Poster-only -- the DICE constants are retained as a
# fallback for the other scripts that still reference them.
ROLE1 = (0.058184, 0.273631, 0.191837)   # dark green  -- the true value
ROLE2 = (0.239379, 0.369361, 0.547850)   # blue
ROLE3 = (0.618071, 0.403068, 0.697898)   # purple
ROLE4 = (0.852116, 0.555525, 0.605505)   # rose        -- the biased/inflated value
ROLE5 = (0.845481, 0.805456, 0.646697)   # sand        -- fills, region bands

# DICE palette (Montana State; retained as a fallback for non-poster scripts).
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
# One role colour per profile shape: roles 1 / 2 / 3 (the strong-contrast lines).
SHAPE_COLOR = {
    "gaussian": ROLE1,
    "voigt": ROLE2,
    "generalized_gaussian": ROLE3,
    "genGauss": ROLE3,
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
        color = _lerp(ROLE2, ROLE4, f)
        ax.plot(x, gaussian(x, amp, 0.0, sig), color=color, lw=2.0)
    ax.annotate("", xy=(3.6, 0.92), xytext=(0.6, 0.92),
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(2.1, 0.96, "time", color="#444", fontsize=9, ha="center")
    ax.text(0.0, 1.02, "early frame:\nnarrow, bright (high CNR)",
            color=ROLE2, fontsize=8, ha="center", va="bottom")
    ax.text(4.6, 0.30, "late frame:\nwide, dim (low CNR)",
            color=ROLE4, fontsize=8, ha="center")
    ax.set_xlabel("position")
    ax.set_ylabel("intensity")
    ax.set_xlim(x[0], x[-1])             # curves run to the frame edges
    ax.set_ylim(0, 1.15)
    # A cartoon: axis labels only, no ticks or numbers.
    ax.set_xticks([])
    ax.set_yticks([])
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
    v = np.linspace(-3.0, 4.0, 700)
    m_true, sd = 0.5, 1.1
    g_un = np.exp(-0.5 * ((v - m_true) / sd) ** 2)     # estimate with no wall
    g = g_un / g_un.max()
    pos = v >= 0
    # The forbidden negative tail, mirrored back across the wall (v -> -v).
    g_ref = np.exp(-0.5 * ((-v - m_true) / sd) ** 2) / g_un.max()
    folded = g + g_ref                                  # fitted-variance pdf

    # The fitted-variance distribution is the original right side plus the
    # reflected negative tail. Draw the reflected tail explicitly (gold dashed,
    # the literal mirror of the left ghost) so the fold reads; the solid curve
    # is their sum, which piles up against the wall and is right-skewed -- it
    # is twice the dashed curves at the wall, not a Gaussian. A light gold fill
    # gives the distribution body.
    axL.fill_between(v[pos], 0, folded[pos], color=ROLE5, alpha=0.45, lw=0)
    axL.plot(v, g, color="#999", ls="--", lw=1.5, label="unbounded estimate")
    axL.plot(v[pos], g_ref[pos], color=ROLE4, ls="--", lw=1.5,
             label="reflected tail")
    axL.plot(v[pos], folded[pos], color=ROLE2, lw=2.4,
             label="fitted variance")
    axL.axvline(0.0, color="k", lw=1.6)

    # The impossible negative tail (ghost), labelled on the empty left side.
    axL.text(-1.65, 0.62, "negative:\nimpossible", fontsize=7.5, color="#777",
             ha="center", va="bottom")
    axL.text(-0.06, 2.08, "wall at 0", fontsize=8, rotation=90, va="top",
             ha="right")

    # True value (centre of the unbounded estimate) and the inflated folded
    # mean -- keyed in the legend so they do not clutter the curve.
    m_folded = (np.trapezoid(v[pos] * folded[pos], v[pos])
                / np.trapezoid(folded[pos], v[pos]))
    axL.axvline(m_true, color=ROLE1, ls=":", lw=1.8, label="true")
    axL.axvline(m_folded, color=ROLE4, ls=":", lw=1.8, label="biased mean")
    axL.set_xlabel("distribution of fitted variances")
    axL.set_ylabel("probability density")
    axL.set_xlim(-3.0, 4.0)
    axL.set_ylim(0, 2.2)
    axL.set_xticks([])
    axL.set_yticks([])
    axL.legend(fontsize=7, loc="upper right", frameon=False)

    # --- Right: the tilted MSD slope --------------------------------------
    t = np.linspace(0, 3.0, 7)
    slope_true = 0.04
    true_line = slope_true * t
    # Per-frame inflation grows as CNR falls (later frames more biased).
    inflation = 0.012 * (t / t.max()) ** 2 + 0.004
    biased = true_line + inflation
    fit = np.polyfit(t, biased, 1)

    axR.plot(t, true_line, color=ROLE1, lw=2.0, label="true slope $\\propto D$")
    axR.plot(t, np.polyval(fit, t), color=ROLE4, lw=2.0,
             label="fitted slope (inflated)")
    axR.plot(t, biased, "o", color=ROLE4, ms=5)
    for ti, tr, bi in zip(t, true_line, biased):
        axR.plot([ti, ti], [tr, bi], color="#bbb", lw=0.8, zorder=0)
    axR.set_xlabel("time")
    axR.set_ylabel("mean-squared displacement")
    axR.set_xticks([])
    axR.set_yticks([])
    axR.legend(fontsize=7, loc="upper left", frameon=False)

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
    cnr0 = np.logspace(np.log10(2.0), np.log10(100.0), 60)
    rel = np.array([np.sqrt(crlb_D(exp, c)) / exp.D for c in cnr0])

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.loglog(cnr0, rel, color=ROLE2, lw=2.2)
    ax.axhline(1.0, color="#999999", ls="--", lw=1.0)
    ax.fill_between(cnr0, 1.0, rel, where=(rel >= 1.0),
                    color=ROLE4, alpha=0.25)
    ax.annotate("a single measurement\ncannot pin D (SD > D)",
                xy=(3.0, 3.0), fontsize=8, color=ROLE3)
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$")
    ax.set_ylabel(r"$\sqrt{\mathrm{CRLB}}\,/\,D$  (best possible SD, units of $D$)")
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
    ax.axhspan(0.95, 1.05, color=ROLE5, alpha=0.45,
               label=r"$\pm 0.05$ band")
    ax.axhline(1.0, color="#666666", lw=1.0)
    for shape, d in shapes.items():
        ax.semilogx(d["cnr"], d["folded"], "o-", ms=4, lw=1.8,
                    color=SHAPE_COLOR.get(shape, ROLE2),
                    label=SHAPE_LABEL.get(shape, shape))
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$")
    ax.set_ylabel(r"corrected  $\langle \hat D\rangle / D_\mathrm{true}$")
    ax.set_ylim(0.7, 1.3)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
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
                    color=SHAPE_COLOR.get(shape, ROLE2),
                    label=SHAPE_LABEL.get(shape, shape))
    ax.set_xlabel(r"per-profile contrast-to-noise  CNR")
    ax.set_ylabel(r"$|$residual$|$  (in $\sigma_0^2$)")
    ax.set_xlim(0.2, 100)
    ax.set_ylim(0, 1.4)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    _style(ax)
    fig.tight_layout()
    path = OUTPUT / "poster_panel5_floor.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure_fftcnr(npz_path: Path) -> Path | None:
    """Panel 4 (CNR-centric spine): fft-cnr recovers CNR across profile shapes.

    Restyles the sibling fft-cnr-paper case-11 sweep (generalized-Gaussian fit
    path -- the one weighting-wall uses) into the poster palette: the recovery
    ratio (estimated / true CNR) vs true CNR, median + interquartile band, one
    line per shape, centred on 1.0. The three poster-primary shapes carry roles
    1-3; the flat-top is a neutral secondary reference.
    """
    if not npz_path.exists():
        print(f"  (fft-cnr case-11 data not found at {npz_path} -- skipping "
              "Panel 4 fft-cnr)")
        return None
    d = np.load(npz_path, allow_pickle=True)
    true_cnr = d["true_cnr"]
    shapes = list(d["shapes"])
    # Map the case-11 shapes to poster labels/colours.
    style = {
        "Gaussian (p=2)": ("Gaussian", ROLE1),
        "heavy-tail (p=1.3)": ("gen-Gaussian (heavy tail)", ROLE3),
        "Voigt": ("Voigt", ROLE2),
        "flat-top (p=4)": ("gen-Gaussian (flat top)", "#888888"),
    }

    # Plot the recovery ratio (estimated / true) rather than estimated vs true:
    # on the diagonal plot every shape collapses onto the 1:1 line and the
    # shape-dependent spread is invisible. The ratio centres on 1.0 and pulls the
    # interquartile bands apart, so the accuracy across shapes reads at a glance.
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    lo, hi = true_cnr.min() * 0.85, true_cnr.max() * 1.15
    ax.axhline(1.0, color="#666666", ls="--", lw=1.0, label="exact recovery")
    for i, shape in enumerate(shapes):
        gg = d[f"gg_{i}"]                       # (n_cnr, n_trials)
        ratio = gg / true_cnr[:, None]          # estimated / true, per trial
        med = np.nanmedian(ratio, axis=1)
        q1 = np.nanpercentile(ratio, 25, axis=1)
        q3 = np.nanpercentile(ratio, 75, axis=1)
        label, color = style.get(shape, (shape, "#333333"))
        ax.fill_between(true_cnr, q1, q3, color=color, alpha=0.15)
        ax.semilogx(true_cnr, med, "o-", ms=4, lw=1.7, color=color, label=label)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"true contrast-to-noise  CNR")
    ax.set_ylabel(r"recovered CNR / true CNR")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    _style(ax)
    fig.tight_layout()
    path = OUTPUT / "poster_panel4_fftcnr.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# Case-11 shape name -> (poster label, role colour); shared by the fft-cnr figures.
_FFTCNR_STYLE = {
    "Gaussian (p=2)": ("Gaussian", ROLE1),
    "heavy-tail (p=1.3)": ("gen-Gaussian (heavy tail)", ROLE3),
    "flat-top (p=4)": ("gen-Gaussian (flat top)", "#888888"),
    "Voigt": ("Voigt", ROLE2),
}


def figure_fftcnr_white_vs_shot(white_npz: Path, shot_npz: Path) -> Path | None:
    """Compare fft-cnr CNR recovery under white vs shot (Poisson) noise.

    Two panels share the CNR axis. Under white noise the recovery ratio sits at
    1.0 (the noise is spatially uniform, so the spectral noise estimate equals
    the peak noise). Under shot noise the variance follows the local signal and
    concentrates under the peak, so fft-cnr's window-averaged noise sits below
    the peak noise and the reported CNR overestimates the true peak SNR by a
    profile-dependent factor (~2.6-3.2x). The split is the subject of fft-cnr
    issue #1; here it is shown, not corrected.
    """
    if not (white_npz.exists() and shot_npz.exists()):
        print("  (white and/or shot fft-cnr sweep missing -- skipping comparison)")
        return None

    fig, (axW, axS) = plt.subplots(2, 1, figsize=(4.6, 5.4), sharex=True)
    for ax, npz, title in ((axW, white_npz, "white noise"),
                           (axS, shot_npz, "shot noise (Poisson)")):
        d = np.load(npz, allow_pickle=True)
        true_cnr = d["true_cnr"]
        ax.axhline(1.0, color="#666666", ls="--", lw=1.0)
        for i, shape in enumerate(list(d["shapes"])):
            ratio = d[f"gg_{i}"] / true_cnr[:, None]
            med = np.nanmedian(ratio, axis=1)
            q1 = np.nanpercentile(ratio, 25, axis=1)
            q3 = np.nanpercentile(ratio, 75, axis=1)
            label, color = _FFTCNR_STYLE.get(shape, (shape, "#333333"))
            ax.fill_between(true_cnr, q1, q3, color=color, alpha=0.15)
            ax.semilogx(true_cnr, med, "o-", ms=3.5, lw=1.6, color=color,
                        label=label)
        ax.set_ylabel("recovered / true CNR")
        ax.text(0.03, 0.93, title, transform=ax.transAxes, fontsize=9,
                va="top")
        _style(ax)
    axW.set_ylim(0.8, 1.2)
    axS.axhline(0.0, color="none")          # keep the y=1 reference in view
    axS.text(0.03, 0.07, "fft-cnr reads peak/(window-averaged noise),\n"
             "not the peak SNR -- a ~3x offset", transform=axS.transAxes,
             fontsize=7.5, va="bottom", color="#555")
    axS.legend(fontsize=7, loc="center right", frameon=False)
    axS.set_xlabel(r"true contrast-to-noise  CNR")
    fig.tight_layout()
    path = OUTPUT / "fftcnr_white_vs_shot.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    print("Producing poster figures...")
    print(f"  Panel 1 schematic (measurement):   {figure_schematic_measurement().name}")
    print(f"  Panel 2A schematic (wall + fold):  {figure_schematic_wall_fold().name}")
    print(f"  Panel 4 (precision floor):         {figure_crlb().name}")
    p4c = figure_fftcnr(FFTCNR_NPZ)
    if p4c:
        print(f"  Panel 4 (fft-cnr CNR recovery):    {p4c.name}")
    p5 = figure_corrected_bias(OUTPUT / "dense_calibration_and_test.txt")
    if p5:
        print(f"  Panel 5 (corrected bias):          {p5.name}")
    pf = figure_floor()
    if pf:
        print(f"  Panel 5 inset (per-profile floor): {pf.name}")
    print("Done.")


if __name__ == "__main__":
    main()
