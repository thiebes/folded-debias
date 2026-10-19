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

## Per-frame residual: the CNR-only lookup and the reference offset

An earlier per-frame table at CNR0 = 20 showed the mean |residual| over the
usable band rising after correction (132 -> 525, 89 -> 155, 142 -> 260 nm^2 for
x = 0.25 / 0.5 / 0.75). Two separate effects produced it.

1. **The 1-D lookup mixes widths.** The D-aware calibration is keyed on CNR
   alone, but here sigma^2 grows ~64x over the usable band and var_bias scales
   with sigma^2. The calibration reaches each CNR along its own peak-decay path,
   while the data's CNR also falls as the spot spreads, so one CNR value pools
   calibration frames from sigma^2 = 625 to ~38,000 nm^2, and `np.interp`
   through those unsmoothed points is erratic (2031, 7773, and 529 nm^2 at CNR
   1.71, 1.56, and 1.48 for x = 0.25). The script now defaults to a joint
   (CNR, sigma^2) D-free calibration on the data's pixel grid
   (`cnr_sigma2_calibration`); `daware` reproduces the old table.
2. **The carrier reference is not the noise-free fit.** At CNR 3 to 10, where
   the noise bias is negligible, the naive residual against `var_true` is
   already -50 to -170 nm^2: the Gaussian fit of the noise-free profile sits
   118 to 148 nm^2 below the carrier second moment on average (up to 256 nm^2).
   That is the shape and emission-to-carrier gap, which the noise correction
   does not address, so a residual against `var_true` mixes it in. The clean
   metric for the noise correction is the residual against the noise-free fit.

Against the noise-free fit at CNR0 = 20 (800 seeds), mean |residual| over the
usable band goes 144 -> 70, 94 -> 61, 170 -> 58 nm^2 with the joint calibration
(signed +140 -> +36, +94 -> -6, +170 -> +5), against 144 -> 462, 94 -> 56,
170 -> 211 with the old lookup.

The joint calibration's weak point is its key, not only its budget. Applied
per measurement at each frame's own noisy sigma_hat^2 (with the fixed-point
inversion), the lookup reads near the grid edge at low CNR and over-subtracts:
at x = 0.5, CNR0 = 10, frames between the floor and CNR 2 go from +815 nm^2
(naive) to -1274 nm^2 (600-run grid), while keying the same grid on the true
sigma^2 gives +550 and a per-measurement smoothed key gives +264 (residuals
against the noise-free fit). The smoothed key (`HIGHN_KEY=smoothed`,
`smoothed_key`) is a robust straight line through the measurement's own
sigma_hat^2 above the floor; it assumes sigma^2(t) is linear, which holds for
this bed's normal diffusion, so it is a bed option, not the library default.

With a denser joint grid (40 CNR points, 20 of them below CNR 3, x 26 sigma x
4000 runs; `cal_highn_cnrsig2_g12_n4000_dense.npz`), mean |residual| against
the noise-free fit over the usable band, x = 0.25 / 0.5 / 0.75:

| CNR0 | key | naive -> corrected (nm^2) | signed corrected (nm^2) |
| ------- | ------- | ------- | ------- |
| 20 | per-frame | 144 -> 65, 94 -> 64, 170 -> 72 | +30, -25, -27 |
| 20 | smoothed | 144 -> 70, 94 -> 35, 170 -> 93 | +38, +25, +90 |
| 10 | per-frame | 177 -> 50, 235 -> 453, 270 -> 365 | -19, -431, -338 |
| 10 | smoothed | 177 -> 65, 235 -> 51, 270 -> 110 | +55, +24, +92 |

Short-window D over the usable band, as D_corr / D_ref at CNR0 8 / 15 / 30:
per-frame key 0.85 / 0.99 / 0.99 (x = 0.25), 0.32 / 0.90 / 0.96 (0.5),
0.34 / 0.90 / 0.96 (0.75); smoothed key 1.00 / 0.99 / 0.99, 0.58 / 0.97 / 0.96,
0.63 / 0.98 / 0.97 (naive 1.08 to 1.21 at CNR0 8). So the dense grid with the
smoothed key removes the noise bias per frame and recovers D within 0.96 to
0.99 from CNR0 15; at CNR0 8 the two slower-spreading maps still read low,
because frames near the floor dominate that short window. The committed figure
and caption remain the earlier D-aware result until the key is chosen.
Tables: `br_halide_highn_perframe_n4000[_smoothed]_cnr0{10,20}.txt` (dense
grid), `br_halide_highn_perframe_cnrsig2_cnr0{10,20}.txt` (600-run grid),
`br_halide_highn_perframe_daware_cnr020.txt` (old lookup).

Key sensitivity (`key_sensitivity.py`, 300 seeds per map and CNR0, dense grid;
table `key_sensitivity.txt`). The fits are shared, so the keys differ only in
the lookup; D is short-window D / D_ref with an SE from resampling the seeds.

- From CNR0 15 the key barely matters. Perturbing the exact key by +/-20
  percent moves D by at most 0.004 (|dD/de| <= 0.018), and the smoothed and
  iterated keys give the same D as the exact key to within 0.003. The
  per-frame key is the exception: it leaves a resolved low bias even here
  (0.90 and 0.89 at CNR0 15, 0.96 at 20, for x = 0.5 and 0.75; SE 0.01).
- The trend keys are not accurate keys. The iterated line reads 8 to 20
  percent below the true sigma^2 on average, with a 15 to 39 percent scatter
  per frame and measurement, yet from CNR0 15 that costs at most 0.003 in D.
- At CNR0 8 to 10 the result depends on the key: dD/de is 0.12 to 0.28,
  even the exact key gives 1.00 to 1.13, and the smoothed and iterated keys
  disagree by up to 0.3 (x = 0.5, CNR0 8: smoothed 0.59, iterated 0.89, exact
  1.00), well beyond their SE of about 0.05. The data do not determine D there
  once the correction is applied, so this bed's D should be quoted from CNR0
  15 only.
- The residual 0.96 to 0.98 of D_ref at CNR0 20 to 30 for x = 0.5 and 0.75
  appears with every key and with the naive fit at CNR0 30, so it is the shape
  and emission-to-carrier gap against the carrier reference, not noise.

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
- `br_halide_highn_perframe.py [cnr_sigma2|daware] [CNR0]` -- the per-frame
  residual (with a signed per-frame table) and short-window D, and the poster
  figure. Default calibration is the joint (CNR, sigma^2) one; `daware` is the
  earlier 1-D lookup that made the committed figure.
- `br_halide_highn_perframe.png` -- the poster figure (CNR0 = 10).

## Depends on (in the library, not here)

The `max_var` bound lives in the shared fit so calibration and application use one
implementation:

- `src/folded_debias/fit.py` -- `fit_variance(x, y, max_var=None)`; default
  `None` keeps the historical `span^2`.
- `src/folded_debias/models.py` -- `GaussianModel(max_var=...)` forwards it.

The scripts also reuse two pure helpers (`fit_msd_powerlaw`, `fit_each_row`) from
`experiments/halide-mixing-brI-legacy/br_halide_validation.py`.

## Running

Needs the pshare for the collapsed maps. Pin the BLAS threads so the process pool
does not oversubscribe, and use a modest worker count if the machine is busy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FOLDED_DEBIAS_WORKERS=4 \
  .venv/Scripts/python.exe experiments/halide-mixing-highn/br_halide_highn_perframe.py
```

Outputs (figure, report, cached calibrations) are written to the repo's `output/`
(gitignored); the figure here is a committed snapshot. Set the panel contrast with
the `CNR0` constant in `br_halide_highn_perframe.py` (10 shows the correction
working; >= 15 is the regime where the floored naive fit is already unbiased and
the correction has little to do).

## See also

- `experiments/halide-mixing-brI-legacy/` (the legacy 20k-walker bed).
