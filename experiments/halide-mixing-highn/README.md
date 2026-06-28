# High-N Br-I bed and the width-fit upper-bound finding

A self-contained study folder for two coupled results:

1. A clean, high-statistics Br-I halide-mixing test bed (400,000-walker average),
   and
2. the finding that the per-frame width fit's **upper bound** -- not the
   contrast-to-noise alone -- governs how large the truncation bias appears.

## The finding

The per-frame Gaussian width fit historically bounded the fitted variance to
`[~1 pixel^2, span^2]`, where `span` is the full analysis window. When the output
window is sized far wider than the spot -- here a 3000 nm box around a 25 nm spot
-- that `span^2` ceiling lets a low-contrast fit wander out toward the window
scale on noise. Two consequences:

- the **naive** D is inflated 3-5x at low CNR (most of the apparent bias is the
  loose bound, not the truncation), and
- the bias **calibration** built from the same fit explodes (`var_bias` ~ 1e5
  nm^2 at CNR ~1-2), so the correction over-subtracts catastrophically.

The fix is a physical cap `var_max = (G * sigma0)^2`, with `G` the maximum
credible width growth `sigma_max / sigma0`. For this bed `G = 12` caps sigma at
300 nm, clearing the real ~196 nm late-frame spot with margin while bounding the
noise-wander. The bound must be **identical** in the calibration and the
application fit, or the subtraction is mismatched.

With the bound, the genuine residual bias is modest -- the naive D is inflated
10-20% at low contrast -- and the folded-normal correction removes it cleanly.
The figure below (`br_halide_highn_perframe.png`, at CNR0 = 10) shows, per Br
fraction, the recovered D ratios:

| Br fraction | D_naive / D_true | D_corr / D_true |
| ----------- | ---------------- | --------------- |
| 0.25        | 1.13             | 1.03            |
| 0.5         | 1.19             | 0.96            |
| 0.75        | 1.22             | 1.06            |

The legacy 20k-walker maps did not expose this because their narrow 150 nm window
already bounded the wander. DICE is unaffected: it sizes its field of view
relative to the FWHM, so its effective width bound already scales with the spot.
The problem is specific to analysing fixed wide-window simulation maps with
`span^2`.

## The bed

Marc Melendez's 20k-walker Br-I maps carry Poisson sampling noise that floors the
intrinsic peak CNR ~30. Re-running his C Brownian-dynamics simulator at 400,000
walkers (16 instances of 25,000, averaged) gives a clean ground truth (intrinsic
peak CNR ~50). The 2-D averaged map was collapsed to 1-D on the run machine
(`P:/jthiebes/WIP/BrI_halide_highN/x_<frac>/collapsed_1d.x_<frac>.npz`); the
multi-GB maps stay on the desktop, and only the small collapsed file is read here.

Geometry: 600 bins over a 3000 nm window (5 nm/bin), 601 frames at 5 ps to 3.0 ns,
initial 1-D sigma ~25 nm. The carrier (transport) truth is the particle second
moment `variance.avg.dat / 2`. It is **Fickian** -- alpha 0.997 / 0.993 / 0.990
for x = 0.25 / 0.5 / 0.75, with D = 6265 / 2650 / 2352 nm^2/ns -- D falling with
Br fraction confirms that mixing inhibits transport. (The legacy maps' apparent
subdiffusion was a short-window / narrow-box artifact, not real transport.)

Over the 3 ns window the peak drops ~147x, so most late frames fall below CNR 1
and are excluded by the floor; the correction is tested on the usable band (a
shorter effective window, which is the fair test).

## Contents

- `br_highn_data.py` -- loads the collapsed 1-D profiles, truth from
  `variance.avg.dat / 2`, imposes the tau ~ 1 ns decay envelope.
- `br_halide_highn_validation.py` -- the CNR0 sweep at this geometry; naive vs
  corrected alpha/D. Uses the 1-D D=0 calibration, which **under**-corrects here
  because the width grows 26-61x (the 2-D (CNR, sigma^2) D-free calibration is the
  deployable path; paper scope).
- `br_halide_highn_perframe.py` -- the per-frame mean residual and short-window D
  with the D-aware calibration, and the poster figure. This is the canonical
  result.
- `br_halide_highn_perframe.png` -- the poster figure (CNR0 = 10).

## Depends on (in the library, not here)

The `max_var` bound lives in the shared fit so calibration and application use one
implementation:

- `src/weighting_wall/fit.py` -- `fit_variance(x, y, max_var=None)`; default
  `None` keeps the historical `span^2`.
- `src/weighting_wall/models.py` -- `GaussianModel(max_var=...)` forwards it.

The scripts also reuse two pure helpers (`fit_msd_powerlaw`, `fit_each_row`) from
`experiments/halide-mixing-brI-legacy/br_halide_validation.py`.

## Running

Needs the pshare for the collapsed maps. Pin the BLAS threads so the process pool
does not oversubscribe, and use a modest worker count if the machine is busy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WEIGHTING_WALL_WORKERS=4 \
  .venv/Scripts/python.exe experiments/halide-mixing-highn/br_halide_highn_perframe.py
```

Outputs (figure, report, cached calibrations) are written to the repo's `output/`
(gitignored); the figure here is a committed snapshot. Set the panel contrast with
the `CNR0` constant in `br_halide_highn_perframe.py` (10 shows the correction
working; >= 15 is the regime where the floored naive fit is already unbiased and
the correction has little to do).

## See also

- `docs/STATUS.md`, the 2026-06-24 section.
