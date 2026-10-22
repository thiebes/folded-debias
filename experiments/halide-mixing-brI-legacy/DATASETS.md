# Datasets -- halide-mixing-brI-legacy

Collaborator data (Marc Meléndez). Referenced by pointer only -- never copied
into git, never moved or renamed on the pshare.

## Source

The loader `br_data.py` reads directly from the pshare:

```text
P:/Marc/shared_data/Halide_mixing
```

These are the legacy ~20,000-walker kinetic-Monte-Carlo Br-I halide-mixing
diffmaps (three Br fractions: 0.25, 0.5, 0.75). No local copy is kept in the
repo; the loader reads the pshare at run time.

## Note

The high-statistics 400,000-walker bed (clean Fickian ground truth) and the
width-fit upper-bound finding live in the separate
`experiments/halide-mixing-highn/` set; see its `DATASETS.md`.
