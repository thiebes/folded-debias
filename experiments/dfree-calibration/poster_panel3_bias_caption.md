**Standard fits read D too high at low contrast, and weighting does not fix it.**
Mean ⟨D_est⟩ / D_true against the initial contrast-to-noise CNR₀, for an
unweighted (OLS) MSD-slope fit and an inverse-variance-weighted fit; the dashed
line is the unbiased value. Unweighted fitting overestimates D by up to about
70× near CNR₀ ≈ 2, and inverse-variance weighting — which assumes a symmetric
error — still leaves D several times too high, reaching the truth only above
CNR₀ ≈ 30. Monte Carlo on the reference measurement: σ₀ = 1, τ = 1,
D = 0.02 σ₀²/τ (so L_D / FWHM₀ ≈ 0.085); 12 frames from 0 to t_final = 3τ;
241 pixels over ±6σ₀. Points are decile averages of a dense CNR₀ sweep.
