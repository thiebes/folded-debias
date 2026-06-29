"""Render the bias-curve poster panels from this set's dense-calibration report:
Panel 3 (the problem -- OLS and inverse-variance both fail at low contrast) and
Panel 5 (the fix -- corrected mean(D_hat)/D_true, three shapes).

Both read the same per-shape decile report, `dense_calibration_and_test.txt` in
this folder (the committed snapshot of the dense run). Shared poster primitives
come from `scripts/poster_style.py`. Writes the figures into this folder.

Run:
    .venv/Scripts/python.exe experiments/dfree-calibration/render_bias_panels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "scripts"))  # poster_style

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from poster_style import (ROLE2, ROLE4, ROLE5, SHAPE_COLOR, SHAPE_LABEL,  # noqa: E402
                          parse_dense_deciles, style)

DENSE = HERE / "dense_calibration_and_test.txt"
OUT_BIAS = HERE / "poster_panel3_bias.png"
OUT_CORRECTED = HERE / "poster_panel5_corrected_bias.png"


def build_bias_curves() -> Path | None:
    """Panel 3: <D_est>/D_true vs CNR_0 for the two standard fits (both fail)."""
    if not DENSE.exists():
        print(f"  ({DENSE.name} not found -- skipping Panel 3 curve)")
        return None
    g = parse_dense_deciles(DENSE).get("gaussian")
    if not g:
        print("  (no gaussian deciles parsed -- skipping Panel 3 curve)")
        return None

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.axhline(1.0, color="#666666", ls="--", lw=1.0)
    # Red carries the catastrophic route (OLS), gold the standard-but-still-
    # failing one (inverse-variance) -- both biased relative to the unbiased line.
    ax.loglog(g["cnr"], g["ols"], "o-", ms=4, lw=2.0, color=ROLE4,
              label="unweighted (OLS)")
    ax.loglog(g["cnr"], g["inv_var"], "o-", ms=4, lw=2.0, color=ROLE2,
              label="inverse-variance")
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$")
    ax.set_ylabel(r"$\langle D_\mathrm{est}\rangle\,/\,D_\mathrm{true}$")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT_BIAS, dpi=300)
    plt.close(fig)
    return OUT_BIAS


def build_corrected_bias() -> Path | None:
    """Panel 5: corrected mean(D_hat)/D_true vs CNR_0, folded-normal, three shapes."""
    if not DENSE.exists():
        print(f"  ({DENSE.name} not found -- skipping Panel 5 curve)")
        return None
    shapes = parse_dense_deciles(DENSE)
    if not shapes:
        print(f"  (no decile data parsed from {DENSE.name} -- skipping)")
        return None

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axhspan(0.95, 1.05, color=ROLE5, alpha=0.20, label=r"$\pm 0.05$ band")
    ax.axhline(1.0, color="#666666", lw=1.0)
    for shape, d in shapes.items():
        ax.semilogx(d["cnr"], d["folded"], "o-", ms=4, lw=1.8,
                    color=SHAPE_COLOR.get(shape, ROLE2),
                    label=SHAPE_LABEL.get(shape, shape))
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$")
    ax.set_ylabel(r"corrected  $\langle \hat D\rangle / D_\mathrm{true}$")
    ax.set_ylim(0.7, 1.3)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT_CORRECTED, dpi=300)
    plt.close(fig)
    return OUT_CORRECTED


if __name__ == "__main__":
    print(build_bias_curves())
    print(build_corrected_bias())
