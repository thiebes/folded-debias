# folded-debias

**Folded-normal correction of the low-CNR bias in fitted widths, for
mean-squared-displacement (MSD) diffusion measurements.**

A fitted width cannot be negative. At low contrast-to-noise ratio (CNR), noise
that would push a width fit below zero pushes it into a right-skewed positive
tail instead, so the fitted variance is biased high. The bias carries through
the MSD fit and inflates the diffusion coefficient D, most strongly in the
late, dim frames of a measurement. Inverse-variance weighting corrects the
unequal noise across frames but cannot remove a mean bias present in every
point.

A folded-normal model of the width estimate gives the size of the bias: the
fitted variance exceeds the true one by the estimation-noise variance,
`E[sigma_hat^2] = sigma^2 + s^2`. The correction subtracts a Monte Carlo
calibrated `s^2(CNR)` from each frame before the slope fit, keyed on the
per-frame CNR. The calibration is built at D = 0, so it does not require a
prior value of D. The derivation is in [`docs/DERIVATION.md`](docs/DERIVATION.md).

## Install

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e .[dev]           # core + matplotlib + pytest + fft-cnr
```

The core depends only on NumPy and SciPy. `fft-cnr` (per-frame CNR from
measured profiles) and `matplotlib` (figures) are optional extras. `uv.lock`
records the exact environment that produced the committed results.

## Quick start

```python
import numpy as np
from folded_debias import Experiment, calibrate_d_free, evaluate

cnr0 = np.array([5, 8, 13, 21, 34, 55, 90])
trunc = calibrate_d_free(cnr0, n_runs=300)        # D-free lookup var_bias(CNR)

exp = Experiment(D=0.02)                          # sigma_0^2/tau; L_D/FWHM_0 ~ 0.085
results = evaluate(exp, cnr0, trunc, n_runs=300)

r = results["folded_normal"]
se = np.hypot(r.spread / np.sqrt(300), r.cal_se)  # evaluation + calibration SE
print(r.mean_bias, se)                            # <D_est>/D_ref per CNR_0
```

On a measurement, the per-frame CNR comes from `folded_debias.cnr.per_frame_cnr`
(which wraps `fft-cnr`), the corrected MSD and weights from
`weights.msd_and_weights("folded_normal", ...)`, the slope from `fit.fit_D`, and
the calibration's contribution to the uncertainty from
`weights.calibration_D_se`.

## What the validation shows

- The width fit reflects rather than clips at the lower bound, so the
  folded-normal route applies (`experiments/boundary-diagnostic`).
- With the calibration's own Monte Carlo error included, the corrected D is
  consistent with the reference within two combined standard errors in 136 of
  140 cells (Gaussian, Voigt, and generalized-Gaussian p = 1.5 and p = 4
  profiles; D from 0.005 to 0.1 sigma_0^2/tau; initial CNR 5 to 90). Small D is
  limited by precision, not bias (`experiments/dfree-calibration`).
- The correction holds above a per-frame CNR of about 1 (0.7 to 1.5 depending on
  shape), where the folded-normal model itself breaks
  (`experiments/correction-floor`). With single-exponential decay to
  3 tau, every frame stays above the floor from an initial CNR of about 20.
- It corrects the anomalous exponent of subdiffusive transport as well as D
  (`experiments/subdiffusive-alpha`, `experiments/nc-solid`).

[`docs/REPO_MAP.md`](docs/REPO_MAP.md) lists every experiment set with its
result, and each set's README gives the details and how to run it.

## Profile models

The bias affects any non-negative width estimator, so the profile model is
pluggable (`folded_debias.models`): `GaussianModel`, `VoigtModel` (a diffusing
Gaussian convolved with a fixed Lorentzian point-spread function), and
`GeneralizedGaussianModel` (shape exponent p; p = 2 is the Gaussian). Scripts
take a model spec; name the generalized-Gaussian shape explicitly:

```bash
python experiments/poc-bias-correction/poc_demo.py voigt
python experiments/boundary-diagnostic/diagnose_boundary.py generalized_gaussian:1.5
```

## Tests

```bash
pytest
```

## Layout

| Path | Role |
| ------- | ------- |
| `src/folded_debias/simulate.py` | profile geometry: diffusion, decay, noise |
| `src/folded_debias/noise.py` | noise laws: white, shot, Poisson, read floor |
| `src/folded_debias/models.py` | profile models and `model_from_spec` |
| `src/folded_debias/fit.py` | per-frame width fit; MSD to D linear fit |
| `src/folded_debias/montecarlo.py` | Monte Carlo ensembles |
| `src/folded_debias/truncation.py` | per-frame statistics; `TruncationModel` lookups and their SE |
| `src/folded_debias/calibration.py` | joint (CNR, sigma^2) calibration |
| `src/folded_debias/weights.py` | the four schemes; calibration SE of D |
| `src/folded_debias/validate.py` | calibration and evaluation sweeps |
| `src/folded_debias/cnr.py` | per-frame CNR from measured profiles via `fft-cnr` |
| `experiments/` | one folder per experiment set (see `docs/REPO_MAP.md`) |
| `illustrations/` | schematic figures |
| `scripts/` | shared calibration tooling and figure rendering |

## Related work

- **DICE**: Monte Carlo engine for optical diffusion measurements (Thiebes and
  Grumstrup, *J. Chem. Phys.* 2024).
- **fft-cnr**: single-profile CNR estimation
  ([PyPI](https://pypi.org/project/fft-cnr/)).

## License

MIT, see [`LICENSE`](LICENSE).
