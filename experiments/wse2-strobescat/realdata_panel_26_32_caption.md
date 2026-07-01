# Caption: real-data panel (canonical) -- MoS2/MoSe2 alloy, 26_32/250716

The correction applied to a real stroboSCAT diffusion measurement (Enrique
Arévalo Rodríguez; MoS2/MoSe2 alloy `mixed_tmdc/26_32/250716`), reduced through
Enrique's own pipeline: his per-frame fitted widths, merged sweeps, MSD fit over
t <= 2 ns forced through the origin with inverse-variance weights. Points are his
naive sigma^2 (red, open) and the folded-normal-corrected sigma^2 (blue, filled);
the diffusion coefficients are stated as numbers rather than fit lines so the
comparison is read from the data.

Result: naive D = 0.0338, corrected D = 0.0334 (ratio 0.989) -- a small but real
correction on a clean, high-CNR real measurement, with the corrected points
sitting slightly below the naive at the noisier later frames. This is the
validation role of the real data: the method returns a stable, self-consistent D
and does not distort clean data. On real data the correction is small wherever
the data is trustworthy and grows toward low CNR (see `diagnostics/index.md` and
`realdata_enrique_pipeline.py` across measurements); the bias-magnitude claim is
carried by the simulation recovery demo
(`diagnostics/_frame_ladder/0_simulated_benefit.png`,
`0b_simulated_distribution.png`), where the truth is known.

Units are his sigma^2 (um^2/ns); the pixel->um scale is an open question with
Enrique (Email 9), so the unit-free corrected/naive ratio is the headline.

Rendered by `realdata_enrique_pipeline.py` (measurement `mixed_tmdc_26_32_250716`);
snapshot copied here from `diagnostics/mixed_tmdc_26_32_250716/msd_enrique_pipeline.png`.

Supersedes the earlier WSe2 `Power_var/250224/0.020` candidate
(`figure_candidates/panel6_wse2-realdata_CANDIDATE-*.png`), which is a
batch-offset artifact -- it goes to negative D once the two interleaved sweeps
are baselined correctly (de-parity). Do not use the WSe2 0.020 panel.
