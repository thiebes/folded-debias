# Caption: br_halide_highn_perframe.png

**Figure N.** Per-frame fitted variance σ²(t) for three Br-I halide-mixing maps
(Br fraction (a) 0.25, (b) 0.5, (c) 0.75), on high-statistics (400,000-walker)
data at initial contrast-to-noise CNR₀ = 10. Each point is the mean over 800
noisy realizations; the indigo line is the reference carrier transport, computed
exactly from the simulation without fitting or added noise. The naive fit (red
circles) rises above the reference as the per-frame contrast falls, consistent
with the upward truncation bias of the bounded width fit. The folded-normal
correction (blue squares) tracks the reference across the usable window. Because
the points are ensemble means, the figure shows the bias removed in the mean; the
per-realization scatter remains.

Grey band: frames below the per-frame CNR floor of 1.3, excluded from the D fit.
The band starts earliest in (a), whose map is the least disordered and spreads
fastest by construction, so it dims soonest.

Over the unshaded window the recovered diffusion coefficient, relative to the
reference value D_ref, is:

| panel | naive | corrected |
| -------- | ----- | --------- |
| (a) 0.25 | 1.13 | 1.03 |
| (b) 0.50 | 1.19 | 0.96 |
| (c) 0.75 | 1.22 | 1.06 |

The naive ratios (D_naive/D_ref) all exceed unity; the corrected ratios
(D_corr/D_ref) lie within 0.06 of unity, scattered on both sides. The ratios are
ensemble-mean point estimates over the 800 realizations; no per-ratio uncertainty
is shown.
