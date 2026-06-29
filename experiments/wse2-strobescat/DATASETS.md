# Datasets -- wse2-strobescat

Collaborator data (Enrique Arévalo Rodríguez). Read-only; referenced by pointer
only -- never copied into git, never moved or renamed on the pshare.

## Source

stroboSCAT (`.npy`) diffusion data on the pshare:

```text
P:\Lab\Enrique\lab\MultiChannelData\        # Perovskites, WSe2, mixed_tmdc, QDs
```

The diagnostics read his processed `analysis_[...]_.npy` per-folder (his per-frame
`profiles`, `sigmas`, `time_arr`, `time_fit`). Measurements registered in
`realdata_diagnostics.py` (paths under `P:\Lab\Enrique\lab\MultiChannelData\`):

```text
WSe2\Power_var\250224\0.020          WSe2\Better_power\5MHz\0.022
WSe2\Power_var\250606\0.025          WSe2\Power_var\250722\0.043
mixed_tmdc\comparison_all\0          mixed_tmdc\pureS\250701   (MoS2)
mixed_tmdc\26_32\250716              mixed_tmdc\50_50\250702   (MoS2/MoSe2 alloys)
```

Reduction / fit follow Enrique's own pipeline at
`P:\Lab\Enrique\python_enrique\` (`MultiChannel\strobe_evaluation_v0.py`,
`Diffusion\diff_eval_app.py` -- the t<=2 ns forced-origin inverse-variance D fit
reproduced by `realdata_enrique_pipeline.py`). The `realdata_triage.py` screen
scans every `analysis_*` folder under `P:\Lab\Enrique\lab\`.

All generated outputs now live in-repo under `diagnostics/` (previously
`P:\jthiebes\WIP\realdata_bias_diagnostics`, now retired).

## Notes

stroboSCAT is white / read-floor-dominated, which matches the shipped white
calibration. TPLM/TCSPC data is shot/Poisson and needs a shot-matched
calibration, not the white one -- match the calibration to the technique.
