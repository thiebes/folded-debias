# Datasets -- nc-solid

Collaborator data (Marc Meléndez; Solari et al. maps). Referenced by pointer only
-- never copied into git, never moved or renamed on the pshare.

## Source

The loader `nc_data.py` reads directly from the pshare:

```text
P:/Marc/shared_data/disordered_perovskite_nanocrystals_solids
```

These are the disordered CH3NH3PbBr3 nanocrystal-solid exciton-hopping maps
(small / medium / large; C16 / C12 / C8 ligands, mean NC radius 3.7 / 5.3 /
7.4 nm), runs of the Solari et al. hopping model that Marc made for this work
(2026-10-21). They are not the paper's Fig. 5 runs (the paper used only variance
curves; its deposit, doi:10.21950/A0AB3H, has no maps). The maps are
~50,000-walker histograms tracking position only; the loader imposes decay
`N0*exp(-t/tau)` and (with `denoise=True`) reconstructs the smooth Gaussian per
frame before measurement noise. No local copy is kept in the repo.

## In-repo derived artifact

`calibration_nc_solid_gaussian.npz` (this folder) is the committed
high-statistics var_bias calibration -- the only large derived artifact kept in
git, because the working copy in `output/` is gitignored.
