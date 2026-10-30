# Snow Accumulation Regimes During Nuclear Winter

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23562900.svg)](https://doi.org/10.5281/zenodo.23562900)
<!-- Concept DOI: resolves to the latest Zenodo-archived release of this repo -->

Analysis of snow storage and snowmelt dynamics in major river basins under
simulated nuclear winter conditions (16, 47, and 150 Tg soot-injection
scenarios), based on CESM model output, compared against a control run.

The pipeline:

1. **Preprocessing** — downloads the raw CESM model output, derives
   precipitation and temperature variables, and builds a land mask.
2. **Snow model** — runs a snow accumulation/melt model on each scenario.
3. **Clustering** — DTW k-means clustering of grid-cell snow storage time
   series, used to exclude permanently snow/ice-covered cells (e.g.
   Antarctica, Greenland) from the anomaly analyses.
4. **Global snow analysis** — control-vs-scenario comparisons of snow
   storage (summary plots, spatial anomaly maps, Hovmoeller diagrams).
5. **Basin analysis** — loads HydroBasins catchment boundaries for six
   major river basins (Rhine, Ganges-Brahmaputra-Meghna, Yellow River,
   Tigris-Euphrates, Ob, Mississippi) and converts snowmelt and rain to
   discharge, producing annual-anomaly, monthly, and seasonal-cycle figures.

See the module docstring at the top of `main.py` for more detail on each
step.

## Installation

Dependencies are managed with [UV](https://docs.astral.sh/uv/). The
committed `uv.lock` pins the exact package versions used to produce the
figures in `results/`.

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the project environment with the locked versions
uv sync --locked
```

## Running the pipeline

```bash
uv run python main.py
```

This downloads the CESM model output (~260 MB) from OSF on first run, then
writes all intermediate datasets and figures under `results/`. All other
input data (HydroBasins, Natural Earth, land mask) is included in `data/`.
Expensive steps (the snow model, DTW clustering) are cached, so subsequent
runs skip recomputation where cached output already exists.

All figures, cluster labels and clustering models are committed under
`results/`, so they can be inspected without running the pipeline. Only the
snow model output (`results/<scenario>/snow_model_output_<scenario>.nc`,
>100 MB each) is not committed; `main.py` regenerates it.

## Code organization

- `main.py` — top-level pipeline script; run this to reproduce all figures.
- `src/preprocessing/` — downloading and preprocessing raw model output.
- `src/processing/` — the snow model and time-series clustering.
- `src/postprocessing/` — plotting and basin-level discharge analysis.
- `src/utilities.py` — shared helpers (masking, grid-cell area, time
  handling).
- `data/` — input data; see [`data/README.md`](data/README.md) for sources.
- `results/` — figures, cluster labels and models; see
  [`results/README.md`](results/README.md).
- `style/` — the [ALLFED matplotlib style sheet](https://github.com/allfed/ALLFED-matplotlib-style-sheet)
  used for all figures, stored locally so they stay reproducible.

## Status

This repository is set up from the
[Scientific-Repository-Template](https://github.com/florianjehn/Scientific-Repository-Template).
Maps use the Robinson projection rather than the template's Winkel Tripel
convention.

## Data sources

- CESM model output (control run and 16, 47, 150 Tg soot-injection
  scenarios): Harrison et al. (2026), openly available via the Open
  Science Framework at https://doi.org/10.17605/OSF.IO/E28GQ.
- River basin boundaries: the HydroBASINS dataset (Lehner & Grill, 2013),
  available at https://www.hydrosheds.org/products/hydrobasins.
- Land mask: Natural Earth 1:110m physical land polygons
  (https://www.naturalearthdata.com/downloads/110m-physical-vectors/),
  public domain.

Full references and download links are listed in
[`data/README.md`](data/README.md).

## Citation

See [`CITATION.cff`](CITATION.cff), or cite directly via the DOI above.

## License

[Apache 2.0](LICENSE)

## Linked paper

TBD.
