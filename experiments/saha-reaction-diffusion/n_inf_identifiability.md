# What the n_inf lifetime fit determines, and the truncation bias on De/Df

Joseph Thiebes -- working note, n_inf diffusion analysis. The PDF
(`n_inf_identifiability.pdf`) is the primary version; this markdown holds the
same content.

Two questions came up in the n_inf analysis. This note answers both from the
data.

1. What does the global multi-fluence lifetime fit determine? It pins down three
   rates: how fast excitons recombine (ν), how fast carriers are trapped (λ), and
   how fast trapped charge decays (γ). It does not pin down the free-charge
   bimolecular rate ρ. The Saha equilibrium ties ρ to ν so tightly that the fit
   sees only their sum.
2. Does the measurement (truncation) bias distort the De/Df fit? No. It is
   negligible against the De/Df error bars.

All numbers come from the three n_inf TPLM fluences, Nexc = 5.5e17, 3.1e18,
7.2e19 cm^-3.

## 1. What the global lifetime fit determines

To learn which rates the three decays pin down, we fit all three PL curves at
once with one shared rate set (ν, ρ, λ, γ, and the trap emission ratio q_trap,
the share of the light that comes from trapped charge); only Nexc differs between
the curves. This shared-rate fit reproduces the overall decay shape
(`n_inf_lifetime_global_fit.png`). It under-fits the late-time tail at the two lower fluences.
That tail constrains the trap parameters (γ and q_trap), not the result below, so
we set it aside here.

To find which rates the data determine, we test each rate one at a time. We fix
the rate at a series of values; at each value we re-fit the other rates and
measure how well the fit matches the data. That measure is the sum of squared
residuals (SSR): for each data point we take the gap between the fitted curve and
the data, square it, and add the squares over all points. (We use the logarithm
of the PL, because the decay spans several decades, so we compare ratios rather
than absolute differences.) A small SSR means a good fit. This
recipe (fix one rate, re-fit the rest, read the SSR) is the profile-likelihood
method, the same method used for the De/Df error bars. It uses no random
sampling, so it returns the same answer every run.

We then read each rate from the shape of its SSR. A clear minimum means the rate
is determined; a flat profile means it is not. The width of the minimum sets the
error bar: a value is still acceptable as long as forcing the rate there barely
worsens the fit, and "barely" is set by the size of the data's own scatter. That
scatter is the residual variance, s^2 = SSR_min/(N-p) = 4.2e-3, where N = 600 is
the number of data points and p = 6 the number of fitted rates. The ~1-sigma
interval for a rate is the range over which its SSR rises by no more than s^2
above the minimum (`n_inf_lifetime_profile_ci.png`).

| rate | meaning | result |
| ------- | ------------- | ------------- |
| ν | exciton recombination | determined as a bracket, ~ [0.8, 6]e8; rejects > 1e9 |
| λ | carrier trapping | determined, 4.06e8, 1-sigma [3.96, 4.17]e8 |
| γ | trapped-charge decay | determined, ~ 1e8 (trap lifetime ~ 10 ns) |
| ρ | free-charge bimolecular | not determined (flat profile, 1e-18 to 1e-9) |

ν is bounded, not pinned to a point. Its SSR is a plateau: equally good for any ν
across a wide band, sharply worse below about 8e7 s^-1, and gradually worse
above. So the data place a firm floor under ν and a soft ceiling over it,
rejecting ν > 1e9 s^-1 and fencing it into roughly [0.8, 6]e8 s^-1. We call this a
bracket: a range, not a single value.

λ is different. It has a clean minimum and is the best-determined rate: λ =
4.06e8 s^-1, with a formal 1-sigma interval [3.96, 4.17]e8 s^-1 (half-width 2.6%,
from a fine grid around the minimum). One caution applies here above all. This
interval assumes the residuals are independent. They are not: a smooth decay makes
neighbouring residuals move together, so the true uncertainty on λ is wider than
2.6%.

### Why the fit cannot determine ρ

The reason is exact, not a matter of weak signal, and it follows from the Saha
equilibrium. Two rates are degenerate when the data cannot tell them apart: any
increase in one can be cancelled by a decrease in the other, leaving the fitted
curves identical, so the fit pins only their combination. Here that combination
is ν + ρA, and the cause is a single algebraic fact.

The Saha free-charge fraction x = 2/(sqrt(4n/A + 1) + 1) obeys the identity

    1 - x = x^2 n/A,

at every density (we checked this by algebra, and numerically to 6e-14). Here A
is the Saha density scale, the density at which carriers are half excitons and
half free charge. The identity makes the two recombination channels share one
density shape: the exciton per-carrier rate ν(1-x) equals ν x^2 n/A, a fixed
multiple of the bimolecular per-carrier rate ρ x^2 n, with the ratio ν/(ρA)
constant at all densities (`n_inf_lifetime_saha_collapse.png`). The total recombination then
collapses to one term,

    ν(1-x) + ρ x^2 n = x^2 n (ν/A + ρ),

so the data see only the sum ν + ρA. Changing the fluence cannot separate ν from
ρ, because the Saha coupling gives the two channels the same dependence on
density.

The fit shows this directly (`n_inf_lifetime_rho_tradeoff.png`). When we fix ρ at a series of
values and re-fit the rest, ν tracks ν_eff - ρA to four significant figures, the
fit quality does not change, and λ and γ do not move (they shift by less than
0.01%). The fit holds ν + ρA fixed and lets ν absorb the change. This continues
until ρA reaches ν (near ρ = 1.5e-9 cm^3/s). There ν can no longer compensate,
and the fit breaks. Only ν trades against ρ; the trap channel is not involved.

Free charges are not absent: the free-charge fraction at t = 0 is 0.58, 0.31, and
0.075 across the three fluences. ρ is invisible not because there are no free
charges, but because the Saha coupling makes its channel rise and fall with
density in step with the exciton channel. To separate the two, you need data
where they scale differently. One option is the low-density regime n << A: there
almost all charge is free (x to 1), so the exciton channel vanishes and the
bimolecular channel returns to ~ ρn, and the two no longer track each other.
Another is an independent measurement of the free charges, such as their
diffusion Df.

## 2. The truncation bias does not affect De/Df

The width fit cannot return a negative width. So when the noise is large, the low
side of the fitted-width scatter is cut off and only the high side survives, which
pulls the average width upward. At low contrast-to-noise (CNR) the fitted width is
therefore biased high, and a width that is too large inflates the late-time MSD.
The question is whether that inflated MSD also inflates De/Df.

It does not. The reason is a single mechanism: low CNR causes both the bias and a
low fit weight. The De/Df fit weights each frame by how trustworthy it is, so a
clean frame counts a lot and a noisy frame counts little (this is inverse-variance
weighting, with weight proportional to 1/sigma^2 and sigma the per-frame MSD
uncertainty). A low-CNR frame is at once biased (from the truncation) and noisy
(large sigma), so the very frames that carry the bias barely count. These are not
two independent safeguards; they are one fact.

The table below puts numbers on this. For each frame it lists the per-frame CNR
(measured with [fft-cnr](https://github.com/thiebes/fft-cnr) on the recorded
profiles) next to that frame's weight in the fit, w_i = 1/sigma_i^2, with the weights scaled to sum to one. The bias is
significant only near CNR 1-2. In the two dimmer fluences, a quarter to two-fifths
of the frames fall in that range, but their summed weight is below 2e-5 of the
total. The weight-weighted mean CNR is 63-268, so the fit is carried by high-CNR
frames, where there is no bias.

| fluence (µJ/cm^2) | CNR range | frames with CNR < 5 | weight share of those frames | weighted-mean CNR |
| ------- | ------------- | ------------- | ------------- | ------------- |
| 13.11   | 5.4--336      | none          | ---           | 268 |
| 2.298   | 1.8--137      | ~ 1/4         | 1.9e-5        | 85  |
| 301.4   | 1.2--93       | ~ 2/5         | 1.6e-5        | 63  |

A simple bound follows. In a weighted average, a group of points that holds weight
fraction w can move the result by at most w times how wrong it is. The bias-prone
frames here hold w < 2e-5. Even if their widths were wrong by an amount as large
as the MSD itself (a deliberate worst case), they would move D by less than 2e-5
of its value. The reported De = 0.030 +/- 0.026 cm^2/s has a relative error near
0.87. So the shift from the bias is about 4e4 times smaller than the error bar.

No truncation correction is needed on the diffusion fits. The limit on De/Df is
the degeneracy between De and Df (the large error bars), which the
profile-likelihood error analysis addresses, not the bias.

## 3. What to report

- Report ν, λ, and γ from the lifetime fit; the data determine them.
- Do not fit ρ from the lifetime data. Fix it from the diffusion data or the
  literature, or report it as unconstrained by the lifetime. Once ρ is fixed, the
  degeneracy is broken and ν is determined, so you can leave ν free rather than
  fixing it too.
- Do not add a Meitner-Auger term (-C n^3). We tested it: its coefficient runs to
  zero and the fit quality does not change with it on or off, so the lifetime
  decays do not constrain it (the same kind of degeneracy as ρ). Keep its bound
  near the literature value (C ~ 1e-28 cm^6/s for a lead halide), but do not trust
  a fitted value that sits at a bound or whose profile is flat.
- Choose the bounds for λ, γ, and C from physical ranges, wide enough that the
  best value sits inside. After fitting, check that no parameter sits at its bound:
  a value pinned at a bound is set by the bound, not the data. γ showed this. A
  bound placed at its best value made γ look pinned, and widening the bound fixed
  it.

## Method

Lifetime identifiability: the shared-rate global fit, then a profile of each rate
(fix one, re-fit the rest with a local optimiser, read the SSR against the fixed
rate). Bias: the per-frame CNR from fft-cnr on the recorded profiles, against the
De/Df fit weight 1/sigma_i^2 (the fit minimises chi^2 = sum_i [(MSD_sim_i -
MSD_i)/sigma_i]^2). The analysis script `lifetime_identifiability.py` regenerates
every figure here.
