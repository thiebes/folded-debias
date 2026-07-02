# fft-cnr-recovery

**Purpose** -- the poster's pivot panel (Panel 4): show that the per-frame CNR the
correction consumes is itself measurable, by recovering the true CNR from noisy
profiles with the open-source fft-cnr estimator.

**Status** -- validated / poster. The underlying validation is the sibling fft-cnr
paper's case 1; this set only restyles that result to the poster palette.

**Contents**

- `build_fft_cnr_recovery_figure.py` -- reduces the sibling repo's full-trial array
  to a small per-(N, CNR) summary and plots the recovered-vs-true CNR hero in the
  Spanish-flag palette (plot_style.md).
- `fft_cnr_recovery_summary.npz` -- committed small derived artifact: per-(N, CNR)
  median and 25/75, 5/95 percentiles over the 1000 trials (~120 floats).
- `fft_cnr_recovery.png` + `fft_cnr_recovery_caption.md` -- the figure and caption.

**Data** -- reduced from the sibling fft-cnr validation suite; the full array is not
committed. See [`DATASETS.md`](DATASETS.md).

**Run**

```bash
.venv/Scripts/python.exe experiments/fft-cnr-recovery/build_fft_cnr_recovery_figure.py
```

No env vars needed (it plots a committed summary; it only touches the source array
when the summary is absent).

**Key result + figure** -- the fft-cnr estimate sits on the 1:1 line from CNR = 2 to
100 across window sizes N = 128-1024 (worst case a slight underestimate at N = 128,
CNR = 100). Figure `fft_cnr_recovery.png`, caption `fft_cnr_recovery_caption.md`.

**Depends on** -- no in-repo helpers; reads the sibling repo's result array only to
build the committed summary.

**See also** -- `docs/poster/POSTER_PLAN.md` (Panel 4); the sibling `fft-cnr-paper`
validation case 1.
