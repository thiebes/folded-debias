# Datasets

Collaborator data, referenced by pointer only -- never copied into the repo,
never moved or renamed on the share.

## n_inf transient PL microscopy (Antonella Cutrupi)

Per-fluence PL decays and MSD for the n_inf sample (3D bulk), used by
`Lifetime_fit_global.py`, `Lifetime_fit_sim_traps.py`, and
`lifetime_identifiability.py`.

Location (read-only):

    P:\Marc\shared_data\data_links\Diffusion_simulation_Antonella\TPLM\
      *_uJcm2.npy                 per-fluence dicts: Time (ns),
                                  Integrated_amplitudes (PL decay), Profiles,
                                  MSD_values(um2), MSD_error (um2)
      Samples_Eb_nu_rho_mu.json   per-sample Eb (meV) and mu (m0)

Point `lifetime_identifiability.py --data` at that TPLM folder.

The write-up built from this data lives in this folder
(`n_inf_identifiability.pdf`, with its LaTeX source and a markdown copy).
