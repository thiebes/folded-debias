# folded-debias

**CNR-aware weighting for MSD diffusion fits — correcting the
truncation-at-zero bias in fitted Gaussian widths.**

A fitted Gaussian width cannot be negative. At low contrast-to-noise ratio
(CNR), the sampling distribution of a fitted width therefore piles up against
that wall at zero: it is left-truncated and right-skewed. That asymmetry leaks
through the mean-squared-displacement (MSD) fit and **inflates the extracted
diffusion coefficient D** — worse as CNR falls.

Standard inverse-variance weighting corrects for heteroscedasticity but assumes
a *symmetric* per-frame error distribution, so it cannot remove this bias. This
project derives and validates a weighting scheme that takes the **per-frame
CNR** as input and corrects for the truncation as well.

> Status: **proof of concept.** The Monte Carlo harness and the candidate
> weight run end to end; the final analytical weight form is still being
> derived. See [`PLAN.md`](PLAN.md).

## Why it matters for nanocrystals

Disordered perovskite nanocrystal solids show exciton diffusion lengths of tens
of nanometres against a diffraction-limited spot — exactly the low-CNR,
small-L_D/FWHM regime where this bias is largest. The bias also contaminates
the *subdiffusive* exponent α extracted from non-Gaussian NC-solid transport,
not just D. Prepared for the Gordon Research Seminar and Conference (GRS/GRC)
on Colloidal Semiconductor Nanocrystals (2026).

## Install

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e .[dev]           # core + matplotlib + pytest + fft-cnr
```

Core dependencies are only NumPy and SciPy. `fft-cnr` (per-frame CNR from real
data) and `matplotlib` (figures) are optional extras.

## Quick start

```python
import numpy as np
from folded_debias import Experiment, calibrate, evaluate

exp = Experiment(D=0.02)                      # short-L_D nanocrystal regime
cnr0 = np.array([2, 3, 5, 8, 12, 20, 40])

model = calibrate(exp, cnr0)                  # Monte Carlo truncation model
results = evaluate(exp, cnr0, model)          # bias curve per weighting scheme

for s, r in results.items():
    print(s, r.mean_bias)                     # <D_est>/D_true vs CNR_0
```

## Profile models

The truncation bias afflicts any non-negative width estimator, so the profile
model is pluggable (`folded_debias.models`): `GaussianModel`, `VoigtModel`
(diffusing Gaussian + fixed Lorentzian PSF — the form Prins' group fits), and
`GeneralizedGaussianModel` (tunable shape exponent — the `fft-cnr` family, the
handle for non-Gaussian transport). Scripts take a model name argument:

```bash
python scripts/poc_demo.py voigt
python scripts/diagnose_boundary.py generalized_gaussian
```

## Proof-of-concept demo

```bash
python scripts/poc_demo.py            # gaussian (default)
```

Prints a bias table and writes `output/poc_demo.png` with two panels:

- **(a)** truncation fraction *p* and MSD skewness vs per-frame CNR — the wall
  distorting the width estimator;
- **(b)** ⟨D_est⟩/D_true vs CNR₀ for OLS, inverse-variance, and the proposed
  CNR-aware weighting.

## Tests

```bash
pytest
```

## Layout

| Path | Role |
|------|------|
| `src/folded_debias/simulate.py`  | profile geometry: diffusion, decay, noise |
| `src/folded_debias/models.py`    | profile models: Gaussian, Voigt, generalized-Gaussian |
| `src/folded_debias/fit.py`       | per-frame Gaussian width fit; MSD→D linear fit |
| `src/folded_debias/montecarlo.py`| Monte Carlo ensembles (model-pluggable) |
| `src/folded_debias/truncation.py`| truncation fraction p(CNR); `TruncationModel` |
| `src/folded_debias/weights.py`   | OLS / inverse-variance / CNR-aware schemes |
| `src/folded_debias/cnr.py`       | per-frame CNR from real data via `fft-cnr` |
| `src/folded_debias/validate.py`  | calibrate + evaluate → bias curves |
| `scripts/poc_demo.py`             | proof-of-concept figure (`[model]` arg) |
| `scripts/diagnose_boundary.py`    | clipping-vs-reflection diagnostic (`[model]` arg) |
| `scripts/explore_regime.py`       | (D, CNR₀) parameter-space sweep |

## Related work

- **DICE** — production Monte Carlo engine for optical diffusion measurements
  (Thiebes & Grumstrup, *JCP* 2024); eventual home for a validated scheme.
- **TRACE** — methods guide that states this bias as currently unresolved.
- **fft-cnr** — single-profile CNR estimation
  ([PyPI](https://pypi.org/project/fft-cnr/)).

## License

MIT — see [`LICENSE`](LICENSE).
