Outputs of `main.py`. Figures, cluster labels and k-means models are committed
so they can be inspected without running the pipeline.

| Path | Content |
|---|---|
| `<scenario>/snow_model_output_<scenario>.nc` | Snow model output (`snow_storage`, `snow_melt`, `rain`) on the CESM grid; `<scenario>` is `control`, `16Tg`, `47Tg` or `150Tg`. **Not committed** (>100 MB each); `main.py` regenerates them from the input data. |
| `47/global_analysis/` | Global snow storage figures (16/47/150 Tg vs. control) |
| `47/basin_analysis/` | River basin discharge figures (47 Tg vs. control) |
| `47/clustering/`, `Control/clustering/` | DTW k-means clustering of snow storage time series |
| `.../clustering/elbow/` | Elbow method checkpoints (labels, models, inertias, plot) |
| `.../clustering/models/` | Final cluster labels and model used in the analysis |
