# Repository map

Every experiment set is one folder under `experiments/`, with its scripts,
result tables, figures, captions, and a `README.md`. This page lists each set
with its main result; the set's README gives the details and how to run it.

## Layout

```text
src/folded_debias/   the library
experiments/         one folder per experiment set (below)
illustrations/       one folder per schematic figure (below)
scripts/             shared calibration tooling and figure rendering
docs/                derivation, validation design, calibration how-to, this map
tests/               package tests
output/              regenerated run outputs and cached calibrations (not tracked)
```

## Experiment sets

### Synthetic method beds

| Set | Scripts | Main result | Figure(s) |
| ------- | ------- | ------- | ------- |
| [poc-bias-correction](../experiments/poc-bias-correction/) | `poc_demo.py` | folded-normal mean \|bias-1\| over CNR0 13-90: 0.04-0.06 on Gaussian, Voigt, p = 1.5, p = 4; the CNR0 5-8 cells depend on the calibration draw | `poc_demo*.png` |
| [boundary-diagnostic](../experiments/boundary-diagnostic/) | `diagnose_boundary.py` | the width fit reflects rather than clips at the lower bound, so the folded-normal route applies | `boundary_diagnostic*.png` |
| [dfree-calibration](../experiments/dfree-calibration/) | `validate_d_zero.py`, `dense_calibration_and_test.py`, `calibration_se_check.py`, `weights_check.py`, `slope_fit_check.py`, `render_bias_panels.py` | corrected D consistent with D_ref within 2 combined SE (evaluation and propagated calibration error) in 136 of 140 cells over 4 shapes, D 0.005-0.1, CNR0 5-90; small D is precision-limited (combined SE up to 7.9); the propagated calibration SE matches 16 independent calibrations (ratio 0.98); per-frame width-variance weights, including frame 0, narrow the D spread to 0.50-0.77 of the MSD-point weighting on every shape | `.txt` tables, `poster_panel3_bias.png`, `poster_panel5_corrected_bias.png` |
| [correction-floor](../experiments/correction-floor/) | `per_profile_floor.py`, `analytical_floor.py`, `per_frame_in_diffusion_experiment.py`, `render_floor_panel.py` | per-profile floor CNR 0.7-1.5 by shape (Gaussian 1.16); the folded-normal model breaks near CNR 1 (A = 0.019-0.10 sigma_0^2) | `.txt` tables, `poster_panel5_floor.png` |
| [noise-law](../experiments/noise-law/) | `shot_noise_validation.py`, `read_floor_sweep.py` | the bias is set by the constant (white / read-floor) part of the noise; a calibration must match the noise law | `read_floor_sweep.png`, `fftcnr_white_vs_shot.png` |
| [coverage-interval](../experiments/coverage-interval/) | `coverage_check.py` | the covariance interval under-covers; report a bootstrap or profile interval | `coverage_check.png` |
| [estimator-comparison](../experiments/estimator-comparison/) | `moment_estimator_failure.py`, `unbounded_fit_comparison.py` | removing the lower bound from the width fit does not substitute for the correction | `method_comparison.png`, `unbounded_fit_comparison.png` |
| [subdiffusive-alpha](../experiments/subdiffusive-alpha/) | `subdiffusive_alpha.py` | mean corrected \|alpha - alpha_ref\| over CNR0 21-90: 0.018-0.100 (SE 0.02-0.04) against 0.18-0.30 naive | `subdiffusive_alpha.png` |
| [multiexp-decay](../experiments/multiexp-decay/) | `multiexp_decay.py` | the correction keys on per-frame CNR, not on the decay law | `multiexp_decay.png` |
| [crlb-precision](../experiments/crlb-precision/) | `crlb.py`, `efficiency_check.py`, `render_crlb_panel.py` (+ `precision_limit.md`) | sqrt(CRLB)/D is several times D at low CNR; with the library's per-frame width-variance weighting the corrected D reaches 0.98-1.08 x sqrt(CRLB) | `poster_panel4_crlb.png` |
| [closed-form-scaling](../experiments/closed-form-scaling/) | `linearity_test.py` | the median bias tracks lambda(-s)/s within a band | `linearity_test.png` |
| [fft-cnr-recovery](../experiments/fft-cnr-recovery/) | `build_fft_cnr_recovery_figure.py` (+ committed summary `.npz`) | fft-cnr recovers the true CNR on the 1:1 line from CNR 2 to 100, N = 128-1024 | `fft_cnr_recovery.png` |

### Transport-simulation beds (Marc Meléndez; data on the pshare, see each `DATASETS.md`)

| Set | Scripts | Main result | Data |
| ------- | ------- | ------- | ------- |
| [nc-solid](../experiments/nc-solid/) | `nc_solid_*` (+ loaders `nc_data.py`, `nc_solid_sim.py`) | the naive exponent climbs toward the ballistic limit at low CNR; the corrected exponent recovers the reference 0.88 | pshare; committed `.npz` calibration |
| [mn-trapping](../experiments/mn-trapping/) | `marc_diffmap_validation.py`, `marc_mn_trapping_plan_a.py` (+ loaders) | the correction removes the noise bias; the shape gap needs a forward fit | pshare; local `data/` (not tracked) |
| [saha-reaction-diffusion](../experiments/saha-reaction-diffusion/) | `nonfickian_validation.py`, `forward_fit_validation.py`, `reaction_diffusion_saha.py` | the correction holds on non-Fickian transport; a forward fit recovers the carrier transport | pshare |
| [halide-mixing-brI-legacy](../experiments/halide-mixing-brI-legacy/) | `br_halide_validation.py` (+ loader) | mixing suppresses transport; the correction has little to do over the short window | pshare |
| [halide-mixing-highn](../experiments/halide-mixing-highn/) | `br_halide_highn_*`, `key_sensitivity.py`, `render_perframe_figure.py` (+ loader) | the width-fit upper bound governs the apparent bias; with a joint (CNR, sigma^2) calibration and a measurement-level key, D is recovered to 0.96-0.99 from CNR0 15 and is not determined below it | pshare (collapsed 1-D) |

### Measured data

| Set | Scripts | Main result | Data |
| ------- | ------- | ------- | ------- |
| [wse2-strobescat](../experiments/wse2-strobescat/) | `render_realdata_panel.py`, `realdata_*` | MoSSe alloy `26_32/250716`: corrected/naive D 0.95 over the full range and 0.97 for t <= 2 ns | pshare (Enrique Arévalo Rodríguez) |
| [cspbbr-strobescat](../experiments/cspbbr-strobescat/) | `reduce.py`, `apply_correction.py`, `run_all.py` | end-to-end correction of measured CsPbBr3 diffusion scans | pshare (Upasana) |
| [zhang-qd-control](../experiments/zhang-qd-control/) | `zhang_qd_panel.py` (+ committed figure CSVs) | the published quantum-dot Dfast is robust to the correction (high CNR) | committed CSVs |

## Illustrations

Schematics without data input live under `illustrations/`, one folder per
figure with its build script, figure, and caption. See
[`illustrations/README.md`](../illustrations/README.md).

| Figure | Folder |
| ------- | ------- |
| Measurement schematic (spreading, dimming population; CNR falls per frame) | [measurement-schematic](../illustrations/measurement-schematic/) |
| Wall-and-fold schematic (wall at zero, reflected tail, tilted MSD slope) | [wall-fold-schematic](../illustrations/wall-fold-schematic/) |

## Shared tooling

- `scripts/build_calibration.py`, `scripts/plot_calibration.py`: builder and
  diagnostic plots for a geometry-specific joint calibration.
- `scripts/run_calibration.sh`: generic calibration runner, called by the
  per-set runners. See [`calibration_runs.md`](calibration_runs.md).
- `scripts/poster_figures.py`, `scripts/poster_fftcnr_dense.py`,
  `scripts/poster_style.py`: render the summary figures and hold their shared
  palette and axis style.

## Datasets (pshare, not tracked)

Each collaborator and measured-data set carries a `DATASETS.md` pointing to its
pshare source:

| Set | pshare source |
| ------- | ------- |
| nc-solid | `P:/Marc/shared_data/disordered_perovskite_nanocrystals_solids` |
| mn-trapping | Marc Meléndez's pshare (Mn diffmaps; local `data/` copy) |
| saha-reaction-diffusion | `P:/Marc/shared_data/data_links/Diffusion_simulation` (+ Marc's Saha code) |
| halide-mixing-brI-legacy | `P:/Marc/shared_data/Halide_mixing` |
| halide-mixing-highn | `P:/jthiebes/WIP/BrI_halide_highN/...` (collapsed 1-D) |
| wse2-strobescat | `P:\Lab\Enrique\lab\MultiChannelData\` (+ `P:\jthiebes\WIP\realdata_bias_diagnostics`) |
| cspbbr-strobescat | `P:/Upasana/CsPbBr_SingleCrystal/Sample_1` |
| zhang-qd-control | published-paper data (committed CSVs) |

Collaborator data is referenced by pointer only; it is not copied into the
repository.
