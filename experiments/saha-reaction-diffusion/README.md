# Saha reaction-diffusion test bed

A Python port of Marc Meléndez's Saha reaction-diffusion model, used as a
realistic non-Fickian test bed for the folded-normal truncation correction. The
model is the spatial extension of the Cutrupi et al. (2026) rate model: a single
charge-pair population diffuses, and at every point the Saha equation sets the
free-charge fraction, so excitons and free charges diffuse at different rates and
the effective diffusivity is density-dependent.

The lifetime fit and the identifiability analysis that grew out of this model
(the Antonella and Carlos free-exciton diffusion paper) now live in the separate
`free-exciton-diffusion` repo. Only the correction-validation part is here.

## Conclusion (2026-06-09): not a poster panel

The folded-normal truncation correction is robust on this bed, but the bed does
not carry a poster panel, for two reasons:

- The truncation effect is modest in the trusted CNR band. The clean correction
  demonstration is the single-population NC-solid bed (where emission is
  proportional to the carrier density, so the bias acts cleanly), not this
  multi-population bed.
- The distortion this bed exposes most vividly, the emission-to-carrier
  nonlinearity, is out of scope for the correction (it is a forward-model
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
- The correction over-subtracts below per-frame CNR ≈ 1.3 on this bed (a low-CNR
  calibration-convexity boundary), so the correction floor is ≈ 1.3.

## Files

- `reaction_diffusion_saha.py` — the model port: the forward simulator. Returns
  the emission, population (carrier), and free-fraction maps over time. The other
  scripts import it. (A copy is also the model foundation of the
  `free-exciton-diffusion` repo.)
- `nonfickian_validation.py` — the validation harness: builds regimes and their
  carrier / emission / fit reference α, runs the estimator paths (Gaussian fit
  plus the non-truncated moment estimator), applies the folded-normal correction,
  and flags the CNR floor.
- `diagnose_overcorrection.py` — pins why the folded correction over-subtracts
  below per-frame CNR ≈ 1.3 (compares the measured per-frame fit bias against
  what the calibration subtracts, frame by frame).
- `forward_fit_validation.py` — stage-2 forward-fit validation: recovers the
  carrier transport by fitting the forward model, applying the folded-normal
  correction to the low-CNR frames. Imports the `folded_debias` library.
- `distortion_figures.py` — illustration/diagnostic figures: the two gaps
  (carrier vs emission vs fit α) and the variance-trajectory mechanism. Not
  poster figures (see the conclusion above).
- `nonfickian_gaps.png`, `saha_second_moment_bias.png`, `poster_saha_nonfickian.png`
  — committed figure snapshots for this bed (the diagnostics referenced above).
- Regenerated figures and calibrations are written to the repo-root `output/`
  (gitignored); the committed snapshots in this folder are the current record.

## Pointers

- Validation framework: [`docs/VALIDATION_DESIGN.md`](../../docs/VALIDATION_DESIGN.md)
- The folded-normal correction: [`docs/DERIVATION.md`](../../docs/DERIVATION.md)
- The lifetime and De/Df diffusion-paper work: the `free-exciton-diffusion` repo.
