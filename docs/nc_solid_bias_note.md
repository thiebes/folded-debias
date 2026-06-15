# Noise-induced bias in the measured transport exponent for disordered nanocrystal-solid diffusion maps

Working technical note for the weighting-wall method and the disordered
perovskite nanocrystal-solid simulations (the maps behind Fig. 5 of Solari et
al.). It is intended for the collaboration, not for circulation. Numbers and
figures are produced by [`scripts/nc_solid_alpha_vs_cnr.py`](../scripts/nc_solid_alpha_vs_cnr.py)
(1000 noise realisations per cell); figures are written to `output/`.

## Summary

A bounded width fit followed by a mean-squared-displacement (MSD) slope fit
mis-estimates the transport exponent α when the per-frame contrast-to-noise
ratio (CNR) is low. On the nanocrystal-solid maps the true transport is mildly
subdiffusive (α ≈ 0.88), but under white measurement noise a standard
all-frames fit reads the motion as ballistic (α at the fit ceiling of 2) until
the initial CNR exceeds roughly 50. The bias direction is fixed: it always
raises the apparent exponent. It can therefore hide subdiffusion but cannot
manufacture it, so a measured subdiffusive exponent is robust to this effect.
The size of the bias is set by the constant, signal-independent part of the
noise: under pure photon-limited (shot) noise it nearly vanishes, and a read
floor of a few counts restores it. The simulation fixes the threshold only for
an assumed noise law; the real measurement's regime is set by its detector and
is measurable from the raw frames.

## The maps as transport ground truth

Three maps (`small`, `medium`, `large`) record 20 frames of a normalised
exciton concentration on a 250 × 250 grid over a 1500 × 1500 nm² area
(6 nm per pixel, 1 ns per frame). They track exciton position only, with no
recombination. To model a transient-photoluminescence measurement we impose the
population decay N₀·exp(−t/τ) with a representative thin-film lifetime τ = 7 ns
(per-sample lifetimes 6 / 14 / 18 ns for the C16 / C12 / C8 ligands), and
project each 2-D frame to a 1-D profile by summing over one axis. The decay is a
per-frame scalar, so it sets the amplitude — hence the per-frame CNR — without
changing the spatial second moment.

The maps are histograms of about 50,000 discrete random walkers, so each pixel
carries Poisson sampling noise from the finite walker count — not the smooth
physical density a continuum measurement would image. We recover that smooth
density by fitting a Gaussian to each frame (the de-noising step in the Methods),
and take its second moment as the transport ground truth: the true D and α
follow from a fit to σ²(t). The diffusion length is small relative to the spot
(L_D / FWHM ≈ 0.14), the regime characteristic of nanocrystal solids.

| map | true α | true D (μm²/ns) |
| ------- | ------- | ------- |
| small | 0.883 | 1.46 × 10⁻⁴ |
| medium | 0.884 | 1.52 × 10⁻⁴ |
| large | 0.875 | 1.99 × 10⁻⁴ |

All three are subdiffusive in the ground truth, so the subdiffusive transport is
a property of the simulated disorder, not of any fitting choice.

## The bias under white noise, and the CNR threshold

We add white (constant-standard-deviation) Gaussian noise to the decayed
profiles, fit each frame's width with a bounded Gaussian, form
MSD(t) = σ̂²(t) − σ̂²(0), and fit MSD = 2D·t^α with α bounded to [0.2, 2]. The
initial CNR, CNR₀, is the frame-0 peak height over the noise standard deviation;
as the spot decays and spreads, the per-frame CNR falls below CNR₀.

The reported bias is the exponent of the seed-averaged MSD trajectory, i.e. the
exponent that survives averaging many measurements. Averaging removes random
scatter but not this bias: it is a property of the estimator's mean, not of any
one realisation.

Figure 1 (`output/nc_solid_alpha_vs_cnr.png`) shows the apparent exponent
against CNR₀ for the three maps. Below CNR₀ ≈ 12 the fit sits at the ceiling
α = 2 (ballistic). The true subdiffusive value returns only once the deepest
fitted frame keeps a CNR of about 2 to 3, which here corresponds to CNR₀ ≈ 35 to
50 across the three maps.

| map | CNR₀ at recovery | deepest-frame CNR there | single-shot s.d. of α |
| ------- | ------- | ------- | ------- |
| small | 50 | 3.2 | ±0.59 |
| medium | 50 | 3.2 | ±0.59 |
| large | 35 | 2.2 | ±0.61 |

Recovery is defined as |α_bias − α_true| ≤ 0.05. The last column is the
standard deviation of the exponent across single realisations at the recovery
point; it remains large (a few tenths) until CNR₀ ≈ 100, so a single
low-contrast measurement is imprecise as well as biased.

Three points bound the reading:

- The value α = 2 is the fit's upper bound. The unconstrained fit would return
  α ≥ 2, so "ballistic" here means "pushed to the ceiling and clipped", not a
  measurement of exactly 2.
- The small-diffusion-length regime amplifies the effect. The spatial variance
  grows only from about 0.040 to 0.044 μm² across the run, so even a modest
  late-time bias dominates the slope. A measurement with larger spreading would
  show a smaller fractional effect on α.
- The exponent is more fragile to noise than the diffusion coefficient. The
  exponent needs a deepest-frame CNR of about 2 to 3, above the CNR ≈ 1 at
  which the diffusion coefficient is recoverable, so discarding only the
  sub-CNR-1 frames does not protect α.

The sample lifetime sets how fast the contrast falls and therefore the
threshold. On the `small` map, a 7 ns lifetime requires CNR₀ ≈ 50, but a
14–18 ns lifetime recovers the exponent already at CNR₀ ≈ 20 (deepest-frame CNR
9–12). The longer-lived samples are well protected; the shorter-lived ones are
where the late frames fall into the difficult range.

## The noise law governs the bias

White noise has a constant floor that persists in the low-signal wings of the
profile, where the bounded width fit takes its leverage. Photon-limited (shot)
noise instead follows the signal and fades in the wings. The two laws therefore
differ in how strongly they drive the bias. We compare them on the `small` map
by replacing white noise with shot noise (Poisson counts scaled so the frame-0
peak holds CNR₀² counts, with an optional constant read-noise floor), holding
CNR₀ fixed.

Figure 2 (`output/nc_solid_noise_law.png`) shows the result. Under pure shot
noise (no read floor) the apparent exponent stays near the true value across
the whole range; at the lowest CNR₀ it falls slightly below the truth rather
than above it. A read floor of 3 counts restores the upward bias but mildly,
recovering by CNR₀ ≈ 12; a floor of 10 counts approaches the white worst case.

| noise law | exponent recovered by | behaviour at low CNR₀ |
| ------- | ------- | ------- |
| white (constant floor) | CNR₀ ≈ 35–50 | pinned at the ceiling α = 2 |
| shot, read = 10 counts | CNR₀ ≈ 20–35 | near white |
| shot, read = 3 counts | CNR₀ ≈ 12 | pinned only at the lowest CNR₀ |
| pure shot, read = 0 | CNR₀ ≈ 8 | falls slightly below the true α |

At equal CNR₀ the shot case is more forgiving for two compounding reasons: the
clean wings weaken the bias mechanism, and the shot CNR falls more slowly with
signal (as the square root of the amplitude rather than linearly), so the
deepest frame keeps a higher CNR. The slight dip below the true exponent under
pure shot noise is consistent with the sign change of the single-frame
width-variance bias under photon-limited noise reported in the fft-cnr
validation cases.

This is the established noise-law result for the truncation bias, here confirmed
on the nanocrystal-solid maps with true Poisson sampling: the bias is set by the
contrast against the constant noise floor, not by the total noise. Differential
or pump-probe techniques, which ride on a bright constant pedestal, are
effectively white and fully in the bias regime; direct photon-counting imaging
is closer to the shot end, with its read floor and background setting how much
bias appears.

## Direction, and the reading for the measured exponent

The bias raises the apparent exponent in every case where it acts. It can move a
true α ≈ 0.87 toward 1 or up to the ballistic ceiling, but it cannot move a true
exponent downward (the one exception, the mild dip under pure shot noise, also
does not create subdiffusion where there is none). A measured subdiffusive
exponent is therefore robust to this effect: if anything, residual bias means
the true transport is slightly more subdiffusive than the fit reports.

This separates cleanly into two cases. The simulated subdiffusion is not at
issue, because the maps are noiseless and α ≈ 0.87 is the disorder physics with
no measurement noise. For an experimental exponent, recovering α < 1 in this
fragile regime implies that the measurement stayed in the high-contrast part of
the data — through some combination of bright signal, a fit window that stops
before the contrast falls, discarding the late low-contrast frames, or operating
near the photon-limited regime where the bias is weak. Which of these applies is
not determined by the simulation.

## Limits, and what would settle it

The threshold quoted here is specific to an assumed noise law and to these
small-diffusion-length maps. Two measurable quantities would convert it from a
design rule into a statement about a real measurement:

- The per-frame CNR of the actual data, read from the raw frames, which places
  the measurement on the CNR₀ axis of Figure 1.
- The detector's noise law — its read floor and gain — fitted from the raw
  frames via the photon-transfer relation (variance versus mean). This places
  the measurement between the curves of Figure 2 and determines whether the
  bias regime applies at all.

A practical consequence follows for applying the correction to real data. The
weighting-wall folded-normal correction is calibrated on white noise and keyed
on the spectral CNR. On shot-dominated data both assumptions are off — the
bias-governing CNR is the peak-local member, and a white calibration
over-corrects photon-limited data — so the correct order of operations is to
establish the noise law first and apply the matched correction, rather than
applying the white-calibrated correction blindly.

## Methods

Maps are loaded read-only and processed by
[`mc_experiments/disordered_nc_solid/nc_data.py`](../mc_experiments/disordered_nc_solid/nc_data.py).
Each frame is a histogram of about 50,000 discrete random walkers (the pixel
values are integer multiples of 1/49984), so it carries Poisson sampling noise
from the finite walker count — a simulation artifact, not the smooth physical
density and not the measurement noise under study. Before adding measurement
noise we de-noise each frame: fit a Gaussian (centre, width) to the noiseless
projection and rebuild it with the same integral. This recovers the continuum
density, conserves the population, and makes the peak fall as 1/σ with spreading
(the raw histogram `max` instead rises, biasing the per-frame CNR high). The
de-noised second moment is the transport ground truth. Each per-frame width is
then fit with the bounded Gaussian estimator
([`weighting_wall.fit`](../src/weighting_wall/fit.py)); the MSD uses the fitted
t = 0 width, MSD = σ̂²(t) − σ̂²(0). The exponent and diffusion coefficient come
from a nonlinear least-squares fit of MSD = 2D·t^α with α ∈ [0.2, 2]. The bias
is the exponent of the mean MSD over 1000 noise realisations; the single-shot
standard deviation is taken across per-realisation exponents. Noise laws are the
pluggable models in [`weighting_wall.noise`](../src/weighting_wall/noise.py):
white noise with standard deviation peak₀/CNR₀, and shot noise with Poisson
counts and an optional read floor, whose per-frame CNR is
peak / √(peak + read²). CNR₀ is the frame-0 peak CNR and means the same starting
contrast across noise laws.
