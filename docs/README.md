# docs

Project documentation, grouped by what it relates to. Start with
`REPO_MAP.md` (where everything is).

## Living references (read these first)

| File | What it is |
| ------- | ------- |
| [REPO_MAP.md](REPO_MAP.md) | The inventory: every experiment set, illustration, and dataset pointer, in one hop. |
| [EXPERIMENT_CONVENTIONS.md](EXPERIMENT_CONVENTIONS.md) | The rulebook for adding any experiment or figure (folder layout, figures/captions, pshare data). |
| [PLOT_STYLE.md](PLOT_STYLE.md) | Plot style specifics for this project (points to the user-level style guide). |

## Method / science specification

| File | What it is |
| ------- | ------- |
| [DERIVATION.md](DERIVATION.md) | The folded-normal derivation of the MSD bias and its correction. |
| [VALIDATION_DESIGN.md](VALIDATION_DESIGN.md) | The validation framework: goal, what "truth" means, how to choose the estimator. |
| [VALIDATION_AUDIT.md](VALIDATION_AUDIT.md) | Audit of each test bed against the validation-design framework. |

## Deliverable briefs

| File | What it is |
| ------- | ------- |
| [poster/](poster/) | The GRC poster deliverable: plan, per-panel text, layout, brief, draft, and assets. See [poster/README.md](poster/README.md). |

## Reference surveys and how-tos

| File | What it is |
| ------- | ------- |
| [alpha_fit_weighting_survey.md](alpha_fit_weighting_survey.md) | Literature survey: how the field fits the anomalous-diffusion exponent alpha (weighted vs unweighted). |
| [calibration_runs.md](calibration_runs.md) | General how-to for building a calibration (the Monte Carlo `var_bias` lookup) for any dataset. |

## Collaborator communications

| File | What it is |
| ------- | ------- |
| [email_collaborators.md](email_collaborators.md) | Draft emails to collaborators (poster + fft-cnr). |
| [marc_briefing.md](marc_briefing.md) / [marc_briefing.tex](marc_briefing.tex) | Briefing prepared for Marc Meléndez. |

## Not here

- **Dataset-specific notes** live with their experiment: `nc_solid_bias_note.md`
  and `nc_solid_calibration_rerun.md` in `experiments/nc-solid/`,
  `cspbbr_calibration_run.md` in `experiments/cspbbr-strobescat/`.
- **Citation audit** (`citations_audit.md`) is at the repo root.
