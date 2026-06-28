# Datasets -- wse2-strobescat

Collaborator data (Enrique Arévalo Rodríguez). Read-only; referenced by pointer
only -- never copied into git, never moved or renamed on the pshare.

## Source

stroboSCAT (`.npy`) diffusion data on the pshare:

```text
P:\Lab\Enrique\lab\MultiChannelData\        # Perovskites, WSe2, mixed_tmdc, QDs
```

The locked panel uses the bulk-WSe2 power scan `Power_var/250224/0.020` (three
long `[-10, 0.65]` ns scans merged). Reduction follows Enrique's own pipeline at
`P:\Lab\Enrique\python_enrique\MultiChannel\` (esp. `strobe_evaluation_v0.py`).

The exploratory-diagnostics WIP lives under `P:\jthiebes\WIP\realdata_bias_diagnostics`
(Joseph's own pshare folder).

## Notes

stroboSCAT is white / read-floor-dominated, which matches the shipped white
calibration. TPLM/TCSPC data is shot/Poisson and needs a shot-matched
calibration, not the white one -- match the calibration to the technique.
