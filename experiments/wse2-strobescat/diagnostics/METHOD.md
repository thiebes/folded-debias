# Method notes: real-data truncation-bias diagnostics

One folder per measurement (named by sample/date/power). Inside each: a
per-frame `manifest.csv`, a naive and a corrected MSD plot, an all-frame overlay
`montage.png`, and a `frames/` subfolder with per-frame overlays for the
noisiest (below-floor or lowest-CNR) frames. `index.md` lists every measurement
with its headline numbers.

## Pipeline (per measurement)

1. **Profiles and times.** Read Enrique's processed `analysis_[...]_.npy`: his
   per-frame reduced 1-D `profiles` (sign-fixed-per-delay, line-cut band through
   the spot) and `time_arr(ns)` (physical time). Sort by time. The canonical
   (most-merged) analysis file in a folder is used; partial scan-index files are
   exploration, not separate measurements.

2. **Sign.** The stroboSCAT differential's sign flips frame to frame. Each frame
   is oriented by correlation to a positive reference bump built from the
   brightest early frames (a per-frame min/max test mis-orients the noisy frames
   and scrambles the width series).

3. **Per-frame CNR and noise law.** `fft-cnr` with `estimate_noise_model=True`,
   measured on the spot window (step 4), gives the white/spectral CNR (+ CI), the
   shot-corrected CNR, and the `var = gain*signal + read^2` decomposition. The
   noise law is white when the fitted gain sits at the noise level (the peak shot
   variance stays below the read variance) -- which is the case for these
   stroboSCAT differentials. CNR must be measured on the spot window, not the
   full line-cut: off-spot low-frequency structure (background/fringes) corrupts
   the spectral CNR at low signal (a real peak can read CNR < 1 and a flat frame
   CNR > 3), which then mis-flags frames at the floor.

4. **Naive width.** A bounded Gaussian fit (`fit_variance`) on a centred
   +/- 4 sigma0 window. This is the standard estimator the correction acts on,
   and matches the estimator the calibration models. (Enrique's `sigmas(um)` are
   kept in the manifest for reference but not used for the width: the px->um
   scale is internally inconsistent, so all widths are reported in px^2.) The
   per-point SE uses his scale-free relative fit error.

5. **Correction.** A D-free `var_bias(CNR)` calibration is built at the
   measurement geometry (the +/- 4 sigma0 window, n_pixels, sigma0) and the
   matched noise law, cached under `_calib/`. Corrected sigma^2 = naive sigma^2 -
   var_bias(CNR) * sigma0^2.

6. **MSD and D.** MSD = sigma^2(t) - sigma^2(0); D = slope/2 (weighted LS). Two
   exclusions define the **retained** frames the headline fit uses:
   - **CNR floor (>= 1.3):** below it `var_bias ~ A/CNR^2` exceeds the signal
     variance and the corrected variance goes negative (over-subtraction); those
     frames are kept in the manifest and plotted (flagged) but excluded.
   - **Degraded-tail truncation:** the late tail is cut where the corrected MSD
     *sustainedly* departs the trend. The inverse-variance line is fit over the
     CNR-kept frames; the residual scale is taken from the early frames (first
     40% of the time span, robust 1.4826*MAD) so it reflects the trend's natural
     scatter; a frame is an outlier if `|residual| > 3*scale` (symmetric). The
     series is truncated at the earliest frame that is itself an outlier and
     whose suffix is at least half outliers -- i.e. where the tail stays off
     trend. An isolated early blip does not trigger a cut; a measurement with no
     clean early trend (uniformly high scatter) is not truncated.

   The retained-frame inverse-variance fit is the headline; the all-frames and
   CNR-kept-only fits are reported for reference. The manifest carries both a
   `keep_flag` (CNR floor) and a `retained_flag` (after the tail cut).

## Caveats

- **px->um scale unresolved** (Enrique's `X(um)` is inconsistent with his
  `sigmas(um)`): all widths and D in px^2 and px^2/ns.
- **Even/odd parity oscillation** is present in the per-frame widths; it is
  shown raw, never smoothed. It adds scatter to single-measurement D.
- **Below-floor frames over-subtract** -- this is expected and is why the floor
  exists; the bias lives in exactly these noisy late frames.
- **var_bias MC error** is not yet propagated into the corrected SE (it is
  sub-dominant to the per-frame fit SE at the calibration budget used); the
  corrected SE currently equals the naive fit SE.
- **Shot-noise calibration** is not yet parameterised to a measurement's fitted
  read/gain; only white-noise (stroboSCAT) measurements are correct as-is.
