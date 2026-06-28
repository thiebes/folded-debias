# Datasets -- nc-solid

Collaborator data (Marc Meléndez; Solari et al. maps). Referenced by pointer only
-- never copied into git, never moved or renamed on the pshare.

## Source

The loader `nc_data.py` reads directly from the pshare:

```text
P:/Marc/shared_data/disordered_perovskite_nanocrystals_solids
```

These are the disordered CH3NH3PbBr3 nanocrystal-solid exciton-hopping maps
(small / medium / large; C16 / C12 / C8 ligands) from Solari et al. The maps are
~50,000-walker histograms tracking position only; the loader imposes decay
`N0*exp(-t/tau)` and (with `denoise=True`) reconstructs the smooth Gaussian per
frame before measurement noise. No local copy is kept in the repo.

## In-repo derived artifact

`calibration_nc_solid_gaussian.npz` (this folder) is the committed
high-statistics var_bias calibration -- the only large derived artifact kept in
git, because the working copy in `output/` is gitignored.
