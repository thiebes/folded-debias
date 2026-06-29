"""Shared poster-figure primitives: the palette, axis style, and the parsers for
the dense-calibration and per-profile-floor reports.

These are genuinely shared across poster panels (the colour roles must be
identical on every panel, and the two report parsers are reused by more than one
figure), so they live in one module that the per-experiment render scripts and the
poster assembly import. Figure-specific render code lives with its experiment (or,
for schematics, in illustrations/); only these primitives are shared.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb

# --- Poster palette (Okabe-Ito colourblind-safe set, Madrid-localised names) -
# Five hues that stay distinguishable under common colour-vision deficiencies,
# each legible on white. Roles carry fixed meaning: blue = the true value,
# red = the biased/inflated value (true-vs-biased reads as blue-vs-red); blue /
# gold / green are the three profile shapes; purple is reserved for fills and
# region bands at low alpha.
ROLE1 = "#0072B2"   # mediterranean blue  -- the true value / Gaussian shape
ROLE2 = "#E69F00"   # flag gold           -- fitted-variance curve / Voigt shape
ROLE3 = "#009E73"   # olive green         -- generalized-Gaussian shape
ROLE4 = "#D55E00"   # spanish red         -- the biased/inflated value
ROLE5 = "#CC79A7"   # castilian purple    -- fills, region bands (use low alpha)

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


def style(ax) -> None:
    """Apply the shared axis style: ticks inside the frame, on all four sides."""
    ax.tick_params(direction="in", which="both", top=True, right=True)


def lerp(c0: str, c1: str, f: float) -> tuple[float, float, float]:
    """Linear blend between two hex colours; f in [0, 1]."""
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple((1.0 - f) * a + f * b)


_DECILE_RE = re.compile(
    r"\[\s*([\d.]+),\s*([\d.]+)\]\s+(\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def parse_dense_deciles(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Parse the per-shape decile report (mean(D_hat)/D_true per scheme).

    Returns ``{shape: {"cnr", "ols", "inv_var", "folded"}}`` using the geometric
    midpoint of each CNR_0 decile as the x value. The decile columns are OLS,
    inverse-variance, folded-normal, MAS -- groups 4-7 of ``_DECILE_RE`` -- so the
    Panel 3 (OLS + inverse-variance) and Panel 5 (folded) renders read one parser.
    """
    text = Path(path).read_text()
    shapes: dict[str, dict[str, np.ndarray]] = {}
    current: str | None = None
    cols: dict[str, list[float]] = {}

    def _flush() -> None:
        if current is not None and cols.get("cnr"):
            shapes[current] = {k: np.array(v) for k, v in cols.items()}

    for line in text.splitlines():
        header = re.match(r"=== (\w+)\s", line)
        if header:
            _flush()
            current = header.group(1)
            cols = {"cnr": [], "ols": [], "inv_var": [], "folded": []}
            continue
        m = _DECILE_RE.search(line)
        if m and current is not None:
            lo, hi = float(m.group(1)), float(m.group(2))
            cols["cnr"].append(np.sqrt(lo * hi))     # geometric midpoint
            cols["ols"].append(float(m.group(4)))
            cols["inv_var"].append(float(m.group(5)))
            cols["folded"].append(float(m.group(6)))
    _flush()
    return shapes


def parse_floor(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse (CNR, residual, SE) from a per-profile-floor output file.

    A data row is ``CNR n raw_mean corrected_mean residual SE |z| [flag]``;
    we take columns CNR (0), residual (4), SE (5).
    """
    cnr, resid, se = [], [], []
    for line in Path(path).read_text().splitlines():
        tok = line.split()
        if len(tok) >= 7 and tok[1] == "50000":
            try:
                cnr.append(float(tok[0]))
                resid.append(float(tok[4]))
                se.append(float(tok[5]))
            except ValueError:
                continue
    return np.array(cnr), np.array(resid), np.array(se)
