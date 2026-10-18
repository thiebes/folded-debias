"""Render the per-profile-floor poster panel (Panel 5 inset, Fig F): |residual|
vs per-profile CNR for the three shapes, marking the failing / useful / reliable
regions.

Reads this set's committed `per_profile_floor_{gaussian,voigt,genGauss}.txt`.
Shared poster primitives come from `scripts/poster_style.py`. Writes the figure
into this folder.

Run:
    .venv/Scripts/python.exe experiments/correction-floor/render_floor_panel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "scripts"))  # poster_style

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from poster_style import (ROLE5, SHAPE_COLOR, SHAPE_LABEL,  # noqa: E402
                          parse_floor, save_poster, style)

OUT = HERE / "poster_panel5_floor.png"


def build() -> Path | None:
    files = {
        "gaussian": HERE / "per_profile_floor_gaussian.txt",
        "voigt": HERE / "per_profile_floor_voigt.txt",
        "genGauss": HERE / "per_profile_floor_genGauss.txt",
    }
    present = {k: v for k, v in files.items() if v.exists()}
    if not present:
        print("  (no per_profile_floor_*.txt found -- skipping Fig F)")
        return None

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    # Regime dividers (per-profile CNR, from DERIVATION sec. 9), kept
    # neutral so the shape curves read cleanly.
    for xline in (0.7, 1.0):
        ax.axvline(xline, color="#bbbbbb", ls="--", lw=1.0)
    ax.text(0.42, 1.25, "failing", color="#777", fontsize=8, ha="center")
    ax.text(0.83, 1.25, "useful", color="#777", fontsize=8, ha="center")
    ax.text(6.0, 1.25, "reliable", color="#777", fontsize=8, ha="center")

    for shape, path in present.items():
        cnr, resid, se = parse_floor(path)
        order = np.argsort(cnr)
        ax.semilogx(cnr[order], np.abs(resid[order]), "o-", ms=3, lw=1.6,
                    color=SHAPE_COLOR.get(shape, ROLE5),
                    label=SHAPE_LABEL.get(shape, shape))
    ax.set_xlabel(r"per-profile contrast-to-noise  CNR")
    ax.set_ylabel(r"$|$residual$|$  (in $\sigma_0^2$)")
    ax.set_xlim(0.2, 100)
    ax.set_ylim(0, 1.4)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    style(ax)
    fig.tight_layout()
    save_poster(fig, OUT)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(build())
