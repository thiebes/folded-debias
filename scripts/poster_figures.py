"""Produce the data-driven poster figures from existing results.

Three of the six poster panels carry a data figure that can be built without
re-running the heavy Monte Carlo: this script regenerates them in one place so
the poster has a single, consistent style.

  * Panel 4 -- the precision floor: sqrt(CRLB)/D vs CNR_0. Computed live from
    the Fisher information (cheap: one matrix inverse per CNR_0). Shows that
    even a perfectly unbiased estimator has a standard deviation of several x D
    at low CNR -- a law, not a method failure.
  * Panel 5 -- the correction works: corrected mean(D_hat)/D_true vs CNR_0 for
    the folded-normal scheme across the three profile shapes, parsed from the
    decile report in ``output/dense_calibration_and_test.txt``.
  * Panel 5 inset (Fig F) -- where it works: per-profile residual vs CNR for
    the three shapes, parsed from ``output/per_profile_floor_*.txt``, with the
    working / useful / failing regions marked.

The two hand-drawn schematics (Panels 1 and 2A) and the figures that already
have a rendered .png (boundary diagnostic, OLS/inverse-variance bias curves)
are not produced here.

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

from weighting_wall.simulate import Experiment  # noqa: E402
from precision_limit.crlb import crlb_D  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)

# Consistent style across the three figures.
FWHM0 = 2.3548                       # FWHM / sigma for a Gaussian
SHAPE_LABEL = {
    "gaussian": "Gaussian",
    "voigt": "Voigt",
    "generalized_gaussian": "generalized-Gaussian",
    "genGauss": "generalized-Gaussian",
}
SHAPE_COLOR = {
    "gaussian": "#1f77b4",
    "voigt": "#d62728",
    "generalized_gaussian": "#2ca02c",
    "genGauss": "#2ca02c",
}


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
    ax.loglog(cnr0, rel, color="#333333", lw=2.0)
    ax.axhline(1.0, color="#999999", ls="--", lw=1.0)
    ax.fill_between(cnr0, 1.0, rel, where=(rel >= 1.0),
                    color="#d62728", alpha=0.12)
    ax.annotate("a single measurement\ncannot pin D (SD > D)",
                xy=(3.0, 3.0), fontsize=8, color="#a11")
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$")
    ax.set_ylabel(r"$\sqrt{\mathrm{CRLB}}\,/\,D$  (best possible SD, in units of $D$)")
    ax.set_title(f"Precision floor  ($L_D/\\mathrm{{FWHM}}_0 \\approx {ld_over_fwhm:.3f}$)",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
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
        ax.semilogx(d["cnr"], d["folded"], "o-", ms=4, lw=1.6,
                    color=SHAPE_COLOR.get(shape, "#333"),
                    label=SHAPE_LABEL.get(shape, shape))
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$")
    ax.set_ylabel(r"corrected  $\langle \hat D\rangle / D_\mathrm{true}$")
    ax.set_title("Folded-normal correction across profile shapes", fontsize=10)
    ax.set_ylim(0.7, 1.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
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
    # Regime bands keyed on per-profile CNR (from STATUS / DERIVATION sec. 9).
    ax.axvspan(1.0, 1e3, color="#2ca02c", alpha=0.08)
    ax.axvspan(0.7, 1.0, color="#ff7f0e", alpha=0.10)
    ax.axvspan(1e-3, 0.7, color="#d62728", alpha=0.08)
    ax.text(3.0, 0.9, "reliable", color="#2a7", fontsize=8, ha="center")
    ax.text(0.83, 0.9, "useful", color="#c70", fontsize=8, ha="center",
            rotation=90)
    ax.text(0.45, 0.9, "failing", color="#a11", fontsize=8, ha="center")

    for shape, path in present.items():
        cnr, resid, se = _parse_floor(path)
        order = np.argsort(cnr)
        ax.semilogx(cnr[order], np.abs(resid[order]), "o-", ms=3, lw=1.4,
                    color=SHAPE_COLOR.get(shape, "#333"),
                    label=SHAPE_LABEL.get(shape, shape))
    ax.set_xlabel(r"per-profile contrast-to-noise  CNR")
    ax.set_ylabel(r"$|$residual$|$  (in $\sigma_0^2$)")
    ax.set_title("Where the correction works", fontsize=10)
    ax.set_xlim(0.2, 100)
    ax.set_ylim(0, 1.4)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = OUTPUT / "poster_panel5_floor.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    print("Producing poster data figures...")
    p4 = figure_crlb()
    print(f"  Panel 4 (precision floor):     {p4.name}")
    p5 = figure_corrected_bias(OUTPUT / "dense_calibration_and_test.txt")
    if p5:
        print(f"  Panel 5 (corrected bias):      {p5.name}")
    pf = figure_floor()
    if pf:
        print(f"  Panel 5 inset (per-profile floor): {pf.name}")
    print("Done.")


if __name__ == "__main__":
    main()
