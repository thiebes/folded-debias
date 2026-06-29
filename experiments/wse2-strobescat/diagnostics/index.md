# Real-data truncation-bias diagnostics

Per-measurement diagnostics of the folded-normal correction on Enrique's diffusion scans. Widths in px^2 (the px->um scale is unresolved). D in px^2/ns. The naive and corrected D are the inverse-variance fits over the **retained** frames: the MSD is baselined per sweep (de-parity: each interleaved delay sweep re-zeroed to its own t0), then frames below the per-frame CNR floor are dropped, then the late tail is truncated where the corrected MSD sustainedly departs the trend (k=3 robust, majority-outlier tail; early scale, so isolated early blips do not trigger it). Frames (kept/retained) reports both cuts.

| Measurement | Noise law | Frames (kept/ret) | CNR range | cut at | D naive (ret) | D corrected (ret) | ratio | Folder |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| WSe2_Better_power_5MHz_0p022 | white | 32/30/22 | 0.8-25.3 | 4.0 ns | 22.216 | 22.195 | 1.00 | [WSe2_Better_power_5MHz_0p022](WSe2_Better_power_5MHz_0p022/) |
| WSe2_Power_var_250606_0p025 | white | 22/20/12 | 1.0-44.6 | 1.5 ns | 15.235 | 15.162 | 1.00 | [WSe2_Power_var_250606_0p025](WSe2_Power_var_250606_0p025/) |
| WSe2_Power_var_250722_0p043 | white | 25/24/18 | 1.1-64.8 | 2.8 ns | 18.030 | 18.018 | 1.00 | [WSe2_Power_var_250722_0p043](WSe2_Power_var_250722_0p043/) |
| mixed_tmdc_26_32_250716 | white | 25/24/22 | 0.5-18.1 | 4.8 ns | 1.480 | 1.401 | 0.95 | [mixed_tmdc_26_32_250716](mixed_tmdc_26_32_250716/) |
| mixed_tmdc_comparison_all_0 | white | 26/25/14 | 1.1-19.8 | 1.4 ns | 1.167 | 1.153 | 0.99 | [mixed_tmdc_comparison_all_0](mixed_tmdc_comparison_all_0/) |
| mixed_tmdc_pureS_250701 | white | 25/22/12 | 0.6-19.8 | 1.3 ns | 1.031 | 1.018 | 0.99 | [mixed_tmdc_pureS_250701](mixed_tmdc_pureS_250701/) |
| mixed_tmdc_50_50_250702 | white | 25/24/23 | 1.1-14.9 | 9.8 ns | -0.931 | -1.527 | 1.64 | [mixed_tmdc_50_50_250702](mixed_tmdc_50_50_250702/) |
| WSe2_Power_var_250224_0p020 | white | 28/24/24 | 0.8-33.3 | - | -2.101 | -2.667 | 1.27 | [WSe2_Power_var_250224_0p020](WSe2_Power_var_250224_0p020/) |
