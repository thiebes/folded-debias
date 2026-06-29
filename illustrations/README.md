# illustrations

Hand-built matplotlib **illustrations** (schematics / cartoons), as opposed to the
data-driven figures that live with their experiment under `experiments/`. An
illustration has no Monte Carlo or measured-data input -- it is drawn to explain a
concept, so its render is fast and deterministic.

Same convention as `experiments/`: **one folder per figure**, each holding its
self-contained build script, the figure, and the figure's caption. A figure's
render code lives with the figure, never in a shared scripts area.

## Layout

```text
illustrations/
  <figure-name>/
    build_<figure_name>.py        # self-contained render; writes the figure here
    <figure_name>.png             # the committed figure (tracked via .gitignore negation)
    <figure_name>_caption.md      # the caption, beside the figure
```

Naming: lower-case, hyphen-separated folder; the build script writes its `.png`
into its own folder (not `output/`), so the illustration regenerates in place.
`*.png` / `*.svg` are globally gitignored, but `.gitignore` negates
`illustrations/**`, so a figure committed in its folder is tracked automatically.

## Contents

| Figure | Folder | Poster panel | Build |
| ------- | ------- | ------- | ------- |
| Measurement schematic -- spreading, dimming population; CNR falls frame by frame | [measurement-schematic](measurement-schematic/) | Panel 1 | `build_measurement_schematic.py` |
| Wall-and-fold schematic -- wall at zero, reflected tail, tilted MSD slope | [wall-fold-schematic](wall-fold-schematic/) | Panel 2A (hero) | `build_wall_fold_schematic.py` |

## Run

```bash
.venv/Scripts/python.exe illustrations/measurement-schematic/build_measurement_schematic.py
.venv/Scripts/python.exe illustrations/wall-fold-schematic/build_wall_fold_schematic.py
```

The wall-fold figure needs the sibling `curved-text` package
(`pip install -e ../curved-text`) for its on-curve labels.

## Notes

These two are the matplotlib **drafts** of the poster's hand schematics (Joseph to
redraw by hand for the final poster); a hand-drawn replacement goes in the same
folder. The poster assembly (`scripts/build_poster.py`) pulls each schematic from
its folder here; `scripts/poster_figures.py` imports the build functions so there is
a single source of truth.
