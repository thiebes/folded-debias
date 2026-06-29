# Saha reaction-diffusion test bed

A Python port of Marc Meléndez's Saha reaction-diffusion model and the work
built around it as a realistic non-Fickian test bed for the weighting-wall
truncation correction. The model is the spatial extension of the Cutrupi et al.
(2026) rate model: a single charge-pair population diffuses, and at every point
the Saha equation sets the free-charge fraction, so excitons and free charges
diffuse at different rates and the effective diffusivity is density-dependent.

## Conclusion (2026-06-09): not a poster panel

The folded-normal truncation correction is robust on this bed, but the bed does
not carry a weighting-wall poster panel, for two reasons:

- The truncation effect is modest in the trusted CNR band. The clean
  correction demonstration is the single-population NC-solid bed (where emission
  is proportional to the carrier density, so the bias acts cleanly), not this
  multi-population bed.
- The distortion this bed exposes most vividly, the emission-to-carrier
  nonlinearity, is out of weighting-wall's scope (it is a forward-model
  inversion, not the noise correction) and is not a novel result (it is the
  subject of the Cutrupi model itself).

The lasting value of the work is the validation framework it forced out, plus a
model fix and a calibration-reliability finding:

- The framework in [`docs/VALIDATION_DESIGN.md`](../../docs/VALIDATION_DESIGN.md):
  truth is the carrier distribution's second moment; three distortions sit
  between the recorded profile and the carrier distribution (noise, handled by
  the folded-normal correction; the instrument PSF; the emission-to-carrier
  mapping, handled by the forward model); and two workflows (simulation truth is
  input-derived, experimental truth is recovered by forward fit or analytics).
- The recombination and emission terms are bimolecular in the free population,
  so they carry x², not x (corrected against Marc's authoritative code).
- The correction over-subtracts below per-frame CNR ≈ 1.3 on this bed (a
  low-CNR calibration-convexity boundary), so the correction floor is ≈ 1.3.


## Files

- `reaction_diffusion_saha.py` — the model port: the forward simulator. Returns
  the emission, population (carrier), and free-fraction maps over time. The
  other scripts import it.
- `nonfickian_validation.py` — the validation harness: builds regimes and their
  carrier / emission / fit reference α, runs the estimator paths (Gaussian fit
  plus the non-truncated moment estimator), applies the folded-normal
  correction, and flags the CNR floor. Actively evolving across parallel
  sessions.
- `diagnose_overcorrection.py` — pins why the folded correction over-subtracts
  below per-frame CNR ≈ 1.3 (compares the measured per-frame fit bias against
  what the calibration subtracts, frame by frame).
- `distortion_figures.py` — illustration/diagnostic figures: the two gaps
  (carrier vs emission vs fit α) and the variance-trajectory mechanism. Not
  poster figures (see the conclusion above).
- `forward_fit_validation.py`, `profile_likelihood_demo.py` — stage-2
  forward-fit and profile-likelihood work (recovering the carrier transport by
  fitting the forward model); see each file's docstring.
- `Lifetime_fit_global.py`, `Lifetime_fit_sim_traps.py` — trapped-population
  lifetime fits for the **adjacent Antonella/Carlos diffusion paper**, kept here
  because they share the Saha engine. They are not weighting-wall poster scope and
  are authored for Antonella's environment (a hardcoded macOS data path, no
  `__main__` guard), so they are not import-smoke-tested here; see each file's
  docstring.
- `lifetime_identifiability.py` — identifiability analysis of the global lifetime
  fit (same adjacent paper): the global fit (`fit`), a deterministic profile
  likelihood of each rate (`profile`), the Saha rho-nu trade (`tradeoff`), and the
  Saha exact-proportionality figure (`saha`). Result: nu, the trapping rate lam,
  and the trap decay gamma are determined; rho is not. The degeneracy is exact:
  the Saha identity `(1-x) = x^2 n/A` makes the exciton rate `nu(1-x)` exactly
  proportional to the bimolecular rate `rho*x^2*n` at every density, so the fit
  determines only `nu + rho*A`. Parameterized (`--data`, `--sample`) and runnable;
  point `--data` at the TPLM folder in `DATASETS.md`. `--no-fit` regenerates the
  figures from the cached best fit without re-running the optimiser. Figures
  `n_inf_lifetime_global_fit.png`, `n_inf_lifetime_profile_ci.png`,
  `n_inf_lifetime_rho_tradeoff.png`, `n_inf_lifetime_saha_collapse.png`, each with
  a caption beside it; the full write-up (`n_inf_identifiability.pdf`, with its
  LaTeX source) is on the pshare (see `DATASETS.md`).
- `nonfickian_gaps.png`, `profile_likelihood_De_Df.png`,
  `saha_second_moment_bias.png`, `poster_saha_nonfickian.png` — committed figure
  snapshots for this bed (the diagnostics referenced above).
- Regenerated figures and calibrations are written to the repo-root `output/`
  (gitignored); the committed snapshots in this folder are the current record.

## Pointers

- Validation framework: [`docs/VALIDATION_DESIGN.md`](../../docs/VALIDATION_DESIGN.md)
- The folded-normal correction: [`docs/DERIVATION.md`](../../docs/DERIVATION.md)
- Collaborator data pointers for this folder: [`DATASETS.md`](DATASETS.md).
