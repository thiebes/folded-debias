# REPO_MAP -- where everything is

The living inventory of the repository. Every experiment set is one folder under
`experiments/`, findable here in one hop with its scripts, status, key result,
figure, and data location. The rulebook for adding new work is
[`EXPERIMENT_CONVENTIONS.md`](EXPERIMENT_CONVENTIONS.md); this file is the
inventory of what currently exists.

Status tags: **scope** = poster / paper / control / adjacent / parked;
**maturity** = validated / WIP / preliminary / frozen.

## Layout

```text
src/folded_debias/   core library -- imported by everything; never moved
experiments/          one folder per experiment set (below)
illustrations/        one folder per hand-built matplotlib schematic (below)
scripts/              shared tooling + poster assembly (not per-bed; below)
docs/                 shared docs (DERIVATION, VALIDATION_DESIGN, poster/, ...)
docs/REPO_MAP.md      this index
docs/EXPERIMENT_CONVENTIONS.md   the rulebook for new experiments/figures
archive/              superseded / frozen material (not maintained)
output/               regenerated run outputs + .npz calibrations (gitignored)
tests/                package tests
```

## Experiment sets

### Method / synthetic beds (use `src/folded_debias` + synthetic Experiments)

| Set | Scripts | Scope / maturity | Key result | Figure(s) |
| ------- | ------- | ------- | ------- | ------- |
| [poc-bias-correction](../experiments/poc-bias-correction/) | `poc_demo.py` | poster P2/P5 / validated | folded-normal mean \|bias-1\| over CNR0 13-90: 0.04-0.06 on Gaussian, Voigt, p = 1.5, p = 4 (CNR0 5-8 depend on the calibration draw) ||bias-1\| 0.12 (G), 0.106 (V), 0.096 (GG) | `poc_demo*.png` |
| [boundary-diagnostic](../experiments/boundary-diagnostic/) | `diagnose_boundary.py` | framing ref / validated | fit reflects, not clips -> folded-normal route | `boundary_diagnostic*.png` |
| [dfree-calibration](../experiments/dfree-calibration/) | `validate_d_zero.py`, `dense_calibration_and_test.py`, `calibration_se_check.py`, `weights_check.py` | poster P5 / validated | corrected D consistent with D_ref within 2 combined SE (evaluation + propagated calibration error) in 136 of 140 cells over 4 shapes, D 0.005-0.1, CNR0 5-90; small D is precision-limited (combined SE up to 7.9); propagated calibration SE confirmed by 16 independent draws (ratio 0.98); per-frame slope weights narrow the D spread to 0.71-0.85 at CNR0 21-34; folded is the robust default | `.txt` tables |
| [correction-floor](../experiments/correction-floor/) | `per_profile_floor.py`, `analytical_floor.py`, `per_frame_in_diffusion_experiment.py` | poster P5 inset / validated | reliable per-profile CNR >~ 1, fails < ~0.3 | `.txt` tables |
| [noise-law](../experiments/noise-law/) | `shot_noise_validation.py`, `read_floor_sweep.py` | poster P2-3 framing / validated | bias is a constant-noise-floor phenomenon; match calibration to the law | `read_floor_sweep.png`, `fftcnr_white_vs_shot.png` |
| [coverage-interval](../experiments/coverage-interval/) | `coverage_check.py` | paper / validated | covariance CI under-covers; report bootstrap/profile interval | `coverage_check.png` |
| [estimator-comparison](../experiments/estimator-comparison/) | `moment_estimator_failure.py`, `unbounded_fit_comparison.py` | poster aside/paper / validated | removing the wall does not substitute for the correction | `method_comparison.png`, `unbounded_fit_comparison.png` |
| [subdiffusive-alpha](../experiments/subdiffusive-alpha/) | `subdiffusive_alpha.py` | poster P6 / validated | corrected exponent tracks the true alpha for CNR0 >= 20; per-cell Monte Carlo SE reported in the table | `subdiffusive_alpha.png` |
| [multiexp-decay](../experiments/multiexp-decay/) | `multiexp_decay.py` | caveat/paper / validated | correction keys on per-frame CNR, not the decay law | `multiexp_decay.png` |
| [crlb-precision](../experiments/crlb-precision/) | `crlb.py` (+ `precision_limit.md`) | poster P3/4 / validated | sqrt(CRLB)/D is several x D at low CNR | `poster_panel4_crlb.png` |
| [closed-form-scaling](../experiments/closed-form-scaling/) | `linearity_test.py` | parked/paper | median bias tracks lambda(-s)/s in a band | `linearity_test.png` |

### Collaborator transport beds (Marc Meléndez; data on pshare -- see each `DATASETS.md`)

| Set | Scripts | Scope / maturity | Key result | Data |
| ------- | ------- | ------- | ------- | ------- |
| [nc-solid](../experiments/nc-solid/) | `nc_solid_*` (+ loaders `nc_data.py`, `nc_solid_sim.py`) | poster P6 headline / validated | naive alpha climbs to ballistic; folded recovers nominal ~0.88 | pshare; committed `.npz` calibration |
| [mn-trapping](../experiments/mn-trapping/) | `marc_diffmap_validation.py`, `marc_mn_trapping_plan_a.py` (+ loaders) | paper P6 / validated | folded removes noise bias; shape gap needs forward fit | pshare; local `data/` (gitignored) |
| [saha-reaction-diffusion](../experiments/saha-reaction-diffusion/) | `nonfickian_validation.py`, `forward_fit_validation.py`, `reaction_diffusion_saha.py` | not-a-panel / parked | correction robust; forward fit recovers carrier transport (lifetime/diffusion-paper work now in `free-exciton-diffusion` repo) | pshare |
| [halide-mixing-brI-legacy](../experiments/halide-mixing-brI-legacy/) | `br_halide_validation.py` (+ loader) | coverage / preliminary | mixing suppresses transport; correction inactive (short window) | pshare |
| [halide-mixing-highn](../experiments/halide-mixing-highn/) | `br_halide_highn_*`, `render_perframe_figure.py` (+ loader) | fit-bound finding / paper | width-fit upper bound governs apparent bias; clean Fickian truth | pshare (collapsed 1-D) |

### Real-data beds

| Set | Scripts | Scope / maturity | Key result | Data |
| ------- | ------- | ------- | ------- | ------- |
| [wse2-strobescat](../experiments/wse2-strobescat/) | `render_realdata_panel.py` (canonical, MoSSe panel) + `realdata_*`; `wse2_realdata_panel.py` is the retracted WSe2 panel | poster P6 real / validated | MoSSe alloy `26_32/250716`: corrected/naive D 0.95 (full range) and 0.97 (t <= 2 ns); the earlier WSe2 0.020 panel is retracted (batch-offset artifact) | pshare (Enrique) |
| [fft-cnr-recovery](../experiments/fft-cnr-recovery/) | `build_fft_cnr_recovery_figure.py` (+ committed summary `.npz`) | poster P4 / validated | fft-cnr recovers the true CNR on the 1:1 line from CNR 2 to 100, N = 128-1024 | committed summary; source array in the sibling fft-cnr repo |
| [cspbbr-strobescat](../experiments/cspbbr-strobescat/) | `reduce.py`, `apply_correction.py`, `run_all.py` | adjacent / WIP | end-to-end real-data correction (Antonella/Carlos paper) | pshare (Upasana) |
| [zhang-qd-control](../experiments/zhang-qd-control/) | `zhang_qd_panel.py` (+ committed figure CSVs) | control / validated | Zhang QD Dfast robust to the correction (high CNR) | committed CSVs; `.opju` gitignored |

## Illustrations (hand-built matplotlib schematics)

Pure schematics (no data input) live under `illustrations/`, one folder per figure,
each with its self-contained build script, figure, and caption -- same convention as
`experiments/`. See [`illustrations/README.md`](../illustrations/README.md).

| Figure | Folder | Poster panel |
| ------- | ------- | ------- |
| Measurement schematic (spreading, dimming population; CNR falls per frame) | [measurement-schematic](../illustrations/measurement-schematic/) | Panel 1 |
| Wall-and-fold schematic (wall at zero, reflected tail, tilted MSD slope) | [wall-fold-schematic](../illustrations/wall-fold-schematic/) | Panel 2A (hero) |

`scripts/poster_figures.py` imports these build functions (single source of truth)
and `scripts/build_poster.py` pulls the panels from these folders.

## Shared (not per-bed; stay in place)

- `src/folded_debias/` -- the core library. Unchanged by the reorg.
- `scripts/` -- shared tooling and poster assembly:
  - `build_calibration.py`, `plot_calibration.py` -- geometry-specific calibration
    builder + diagnostic (used by several beds).
  - `run_calibration.sh` -- generic calibration runner (called by per-bed runners).
  - `poster_figures.py`, `poster_fftcnr_dense.py`, `build_poster.py` -- poster
    assembly (renders/builds the poster). See "Poster assembly" below.
  - `legacy/` -- deprecated/superseded exploratory scripts (run with `PYTHONPATH=src`).
- `docs/` -- shared docs (see [`docs/README.md`](README.md) for the grouped
  index): `PLAN.md`, `DERIVATION.md`,
  `VALIDATION_DESIGN.md`, `VALIDATION_AUDIT.md`,
  `citations_audit.md`, `email_collaborators.md`,
  `marc_briefing.md`, `alpha_fit_weighting_survey.md`, `calibration_runs.md`,
  and `docs/poster/`. Dataset-specific notes moved to their experiment folders
  (`nc_solid_bias_note.md`, `nc_solid_calibration_rerun.md` ->
  `experiments/nc-solid/`; `cspbbr_calibration_run.md` ->
  `experiments/cspbbr-strobescat/`).

## Archive

- `archive/` -- superseded material (`carrier_vs_emission.py`,
  `POC_RECOMMENDATION.md`, `REGIME_EXPLORATION.md`) plus the frozen
  `fresh_eyes/` and `fresh_eyes_round2/` independent re-derivations. Not
  maintained; see `archive/README.md`.

## Datasets (pshare, never committed)

Each collaborator/real-data set carries a `DATASETS.md` pointing to its pshare
source. Roll-up:

| Set | pshare source |
| ------- | ------- |
| nc-solid | `P:/Marc/shared_data/disordered_perovskite_nanocrystals_solids` |
| mn-trapping | Marc's pshare (Mn diffmaps; local `data/` copy, gitignored) |
| saha-reaction-diffusion | `P:/Marc/shared_data/data_links/Diffusion_simulation` (+ Marc's Saha code) |
| halide-mixing-brI-legacy | `P:/Marc/shared_data/Halide_mixing` |
| halide-mixing-highn | `P:/jthiebes/WIP/BrI_halide_highN/...` (collapsed 1-D) |
| wse2-strobescat | `P:\Lab\Enrique\lab\MultiChannelData\` (+ `P:\jthiebes\WIP\realdata_bias_diagnostics`) |
| cspbbr-strobescat | `P:/Upasana/CsPbBr_SingleCrystal/Sample_1` |
| zhang-qd-control | published-paper data (committed CSVs; no pshare) |

Collaborator data is referenced by pointer only -- never copied into git, never
moved or renamed on the pshare.

## Poster assembly (in `docs/poster/` and `scripts/`) -- pending follow-ups

The poster assembly stays in `docs/poster/` (text, `POSTER_PLAN`, `POSTER_TEXT`,
layout, `poster_draft.pptx`, QR codes) and `scripts/` (`build_poster.py`,
`poster_figures.py`). Two convention follow-ups are **deferred** (the poster is
under active edit; moving its assets mid-edit risks collisions):

1. **`docs/poster/figure_candidates/` relocation (done).** Each candidate now lives
   in a `figure_candidates/` subfolder of the experiment that produced it (the two
   schematic drafts in their `illustrations/` folder); see
   `docs/poster/figure_candidates/INDEX.md` for the full candidate -> location map.
   Only the `panel4_fftcnr_*` candidates remain in `docs/poster/figure_candidates/`,
   because they are sourced from the sibling `fft-cnr` / `fft-cnr-paper` repos and
   have no folded-debias experiment to live in.

2. **`scripts/poster_figures.py` split (done).** Every panel's render code now
   lives with its figure: the schematics in `illustrations/`, the data panels in
   their experiments (`render_crlb_panel.py` in `crlb-precision`,
   `render_bias_panels.py` in `dfree-calibration`, `render_floor_panel.py` in
   `correction-floor`). `poster_figures.py` is now an orchestrator that imports each
   builder; the only panels it still defines are the two fft-cnr recovery figures,
   which are sourced from the sibling `fft-cnr-paper` repo and have no
   folded-debias experiment to live in. Shared poster primitives (palette, axis
   style, the report parsers) are in `scripts/poster_style.py`. `build_poster.py`
   pulls each panel from its home (only the sibling-sourced fft-cnr panel still comes
   from `output/`).
