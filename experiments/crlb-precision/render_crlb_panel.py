"""Render the CRLB precision-floor poster panel (Panel 4): sqrt(CRLB)/D vs CNR_0
at the poster's working D (L_D/FWHM_0 ~ 0.085).

The figure's render code lives with its experiment. Shared poster primitives
(palette, axis style) come from `scripts/poster_style.py`; the CRLB itself is
`crlb_D` in this folder. Writes the figure into this folder.

Run:
    .venv/Scripts/python.exe experiments/crlb-precision/render_crlb_panel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))      # weighting_wall
sys.path.insert(0, str(HERE.parent.parent / "scripts"))  # poster_style
sys.path.insert(0, str(HERE))                            # local crlb.py

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from weighting_wall.simulate import Experiment  # noqa: E402
from crlb import crlb_D  # noqa: E402  (this folder)
from poster_style import ROLE2, ROLE3, ROLE4, style  # noqa: E402

OUT = HERE / "poster_panel4_crlb.png"


def build() -> Path:
    exp = Experiment()                 # defaults: D = 0.02 sigma_0^2/tau
    cnr0 = np.logspace(np.log10(2.0), np.log10(100.0), 60)
    rel = np.array([np.sqrt(crlb_D(exp, c)) / exp.D for c in cnr0])

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.loglog(cnr0, rel, color=ROLE2, lw=2.2)
    ax.axhline(1.0, color="#999999", ls="--", lw=1.0)
    ax.fill_between(cnr0, 1.0, rel, where=(rel >= 1.0), color=ROLE4, alpha=0.25)
    ax.annotate("a single measurement\ncannot pin D (SD > D)",
                xy=(3.0, 3.0), fontsize=8, color=ROLE3)
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$")
    ax.set_ylabel(r"$\sqrt{\mathrm{CRLB}}\,/\,D$  (best possible SD, units of $D$)")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(build())
