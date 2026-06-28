# Datasets on the pshare -- high-N Br-I bed

Large data for this experiment is not committed to the repo. It lives on the
pshare under `P:\jthiebes\WIP\BrI_halide_highN\` (one subfolder per Br fraction
`x_<frac>`, `<frac>` in `0.25`, `0.5`, `0.75`). The repo reads only the small
collapsed file.

| dataset | pshare path | what it is |
| ------- | ----------- | ---------- |
| collapsed 1-D profiles | `BrI_halide_highN\x_<frac>\collapsed_1d.x_<frac>.npz` | the 400,000-walker averaged map projected to 1-D (prof_x/prof_y + axes + metadata). This is the only file `br_highn_data.py` reads. ~1 MB each. |
| carrier variance truth | `BrI_halide_highN\x_<frac>\variance.x_<frac>.avg.dat` | the particle second moment per frame; the 1-D carrier truth is this / 2. ~30 KB each. |
| averaged 2-D map | `BrI_halide_highN\x_<frac>\diffmap.x_<frac>.avg.npy` | the 600x600x601 averaged density map before the 1-D collapse. ~865 MB each -- on the pshare only, never committed. |
| per-instance raw runs | `BrI_halide_highN\x_<frac>\inst_*\` | the 16 individual 25,000-walker realizations that were averaged. |
| run config | `BrI_halide_highN\x_<frac>\parameters.x_<frac>.dat` | the Brownian-dynamics parameters for this fraction. |

The multi-GB raw simulation outputs (full per-instance diffmaps) remain on the
desktop machine that produced them; only the averaged and collapsed products are on
the pshare, and only the collapsed `.npz` is needed by the analysis here.

Connection to the code: `br_highn_data.py` (`HIGHN = P:/jthiebes/WIP/BrI_halide_highN`)
loads `collapsed_1d.x_<frac>.npz` and `variance.x_<frac>.avg.dat` for each fraction.
