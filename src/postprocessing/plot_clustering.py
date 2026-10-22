import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.processing.clustering import prepare_time_series
import xarray as xr
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import math
from src.utilities import compute_grid_cell_area
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def plot_elbow(
    inertias_df: pd.DataFrame,
    save_dir: str,
    color: str = "#4575b4",
):
    """
    Plot the Elbow method (inertia vs. k).

    Arguments:
        inertias_df:    pd.DataFrame, index "k" - inertia values, as returned by
                        elbow_method
        save_dir:       str - directory where the plot is saved
        color:          str - line/marker color for the plot
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax_elbow = plt.subplots(figsize=(6, 3))

    ax_elbow.plot(inertias_df.index, inertias_df["inertia"], marker="o", color=color)
    ax_elbow.set_xlabel("Number of clusters")
    ax_elbow.set_ylabel("Distortion")
    ax_elbow.set_title("Elbow method")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "elbow_silhouette.png"), dpi=600, bbox_inches="tight"
    )
    plt.close()


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> np.ndarray:
    """
    Compute an area-weighted quantile across cells for each time step.

    Arguments:
        values:   np.ndarray, shape (n_cells, time_steps) - time series values per cell
        weights:  np.ndarray, shape (n_cells,)            - area weight per cell
        quantile: float                                   - desired quantile, e.g. 0.5 for median

    Returns:
        np.ndarray, shape (time_steps,) - weighted quantile value for each time step
    """

    # Sort cells by value independently for each time step
    sorted_indices = np.argsort(values, axis=0)  # (n_cells, time_steps)
    sorted_values = np.take_along_axis(values, sorted_indices, axis=0)
    # Weights must follow the same sort order as their corresponding cell values
    sorted_weights = weights[sorted_indices]  # broadcasts over time steps

    # Build normalised cumulative weight distribution (CDF over area)
    cumulative_weights = np.cumsum(sorted_weights, axis=0)
    cumulative_weights /= cumulative_weights[-1, :]

    # First cell index where the cumulative area fraction reaches the desired quantile
    indices = np.argmax(cumulative_weights >= quantile, axis=0)  # (time_steps,)
    quantile_values = sorted_values[indices, np.arange(values.shape[1])]

    return quantile_values  # (time_steps,)


def plot_cluster_combined(
    da: xr.DataArray,
    timeseries_scenario: np.ndarray,
    timeseries_ctrl: np.ndarray,
    labels: np.ndarray,
    cell_areas: np.ndarray,
    fractions: np.ndarray,
    n_clusters: int = 5,
    title: str = None,
    parameter_name: str = "Snow Storage (mm)",
    save_path: str = "./results/clustering",
):
    """
    Plot a combined figure: spatial cluster map on top, per-cluster time series below.

    Clusters are reordered by ascending mean snow storage. Each time series panel shows
    area-weighted quantile bands for the scenario and the control median for comparison.

    Arguments:
        da:                  xr.DataArray, shape (time, lat, lon) - defines the spatial grid
        timeseries_scenario: np.ndarray, shape (n_land_cells, time_steps) - scenario time series
        timeseries_ctrl:     np.ndarray, shape (n_land_cells, time_steps) - control time series
        labels:              np.ndarray, shape (n_land_cells,)            - cluster assignments
        cell_areas:          np.ndarray, shape (n_land_cells,)            - grid cell area per cell
        fractions:           np.ndarray, shape (n_land_cells,)            - land fraction per cell
        n_clusters:          int - number of clusters
        title:               str - used in the output filename
        parameter_name:      str - label for the y-axis
        save_path:           str - directory where the plot is saved
    """
    # Sort clusters by ascending mean snow storage
    colors_full = ["#f1eef6", "#abd9e9", "#74add1", "#4575b4", "#542788"]
    color = "#4575b4"

    effective_areas = cell_areas * fractions

    # Calculate mean snow storage for each cluster
    cluster_means = {}
    for cluster in range(n_clusters):
        cluster_mask = labels == cluster
        cluster_ts = timeseries_scenario[cluster_mask]
        cluster_means[cluster] = cluster_ts.mean()

    # Sort the clusters by mean snow storage and create a mapping from old to new labels
    sorted_clusters = sorted(cluster_means, key=lambda x: cluster_means[x])
    label_mapping = {old: new for new, old in enumerate(sorted_clusters)}
    labels_remapped = np.array([label_mapping[label] for label in labels])

    # Create a colormap for the clusters
    colors = colors_full[:n_clusters]
    cmap = mcolors.ListedColormap(colors)
    unique_classes = np.arange(n_clusters)
    bounds = np.arange(unique_classes[0] - 0.5, unique_classes[-1] + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, ncolors=n_clusters)

    # Grid Layout for the combined plot
    n_cols = 2
    n_ts_rows = math.ceil(n_clusters / n_cols)
    n_rows = 1 + n_ts_rows

    fig_height = 10 + 4 * n_ts_rows
    fig = plt.figure(figsize=(20, fig_height))

    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        height_ratios=[2.5] + [1] * n_ts_rows,
        hspace=0.35,
        wspace=0.25,
    )

    map_ax = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())

    ts_axes = []
    for i in range(n_clusters):
        row = 1 + i // n_cols
        col = i % n_cols
        ts_axes.append(fig.add_subplot(gs[row, col]))

    for j in range(n_clusters, n_ts_rows * n_cols):
        row = 1 + j // n_cols
        col = j % n_cols
        fig.add_subplot(gs[row, col]).set_visible(False)

    # Create the spatial cluster map
    # Map 1d cluster labels back to the 2d spatial grid using the land mask
    template = da.isel(time=0)
    cluster_map = xr.full_like(template, fill_value=np.nan)
    land_mask = ~np.isnan(da.isel(time=0))
    cluster_map.values[land_mask.values] = (
        labels_remapped  # map cluster labels to the spatial grid
    )

    # Plot the cluster map with coastlines and borders
    cluster_map.plot(
        ax=map_ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=True,
        cbar_kwargs={"label": "Cluster", "ticks": unique_classes},
    )
    map_ax.set_title("")
    map_ax.coastlines(linewidth=0.5, zorder=11)
    map_ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=11)
    map_ax.add_feature(cfeature.OCEAN, color="lightgrey", zorder=10)

    # Time series panels in remapped cluster order (ascending mean snow storage).
    # labels still holds original cluster indices, so we translate new → old before masking.
    for new_cluster_idx in range(n_clusters):
        old_cluster = sorted_clusters[
            new_cluster_idx
        ]  # e.g. new=0 → old=3 (least snow)
        cluster_mask = labels == old_cluster  # cells belonging to this cluster
        cluster_ts = timeseries_scenario[cluster_mask]
        cluster_ts_ctrl = timeseries_ctrl[cluster_mask]
        cluster_weights = effective_areas[cluster_mask]

        ax = ts_axes[new_cluster_idx]  # subplot panel in new display order

        # Draw symmetric quantile bands: Q10–Q90, Q20–Q80, Q30–Q70, Q40–Q60.
        # Higher q → narrower band → stronger alpha → darker fill.
        # q_up_overall tracks the highest upper quantile across all bands and time steps,
        # used afterwards to set a consistent y-axis upper limit.
        q_up_overall = 0
        for q in np.arange(0.1, 0.6, 0.1):
            q_up = weighted_quantile(cluster_ts, cluster_weights, 1 - q)
            q_down = weighted_quantile(cluster_ts, cluster_weights, q)
            q_up_overall = max(q_up_overall, q_up.max())
            ax.fill_between(
                range(cluster_ts.shape[1]),
                q_down,
                q_up,
                color=color,
                alpha=q * 2,
            )
        # Set y-limits based on the widest band (Q10–Q90 upper edge) plus a small margin
        y_offset = -0.03 * (q_up_overall + 100)  # 3% of total range as lower margin
        ax.set_ylim(y_offset, q_up_overall + 100)
        ax.set_ylabel("Snow Storage [mm]")

        median = weighted_quantile(cluster_ts, cluster_weights, 0.5)
        ax.plot(
            range(cluster_ts.shape[1]),
            median,
            color="black",
            linewidth=2,
            label="Median Scenario",
        )

        median_ctrl = weighted_quantile(cluster_ts_ctrl, cluster_weights, 0.5)
        ax.plot(
            range(cluster_ts_ctrl.shape[1]),
            median_ctrl,
            color="#E34444",
            linewidth=1.2,
            linestyle="--",
            label="Median Control",
        )

        cluster_area_fraction = cluster_weights.sum() / effective_areas.sum() * 100
        ax.set_title(
            f"Cluster: {new_cluster_idx}, {cluster_area_fraction:.1f}% of Land Area",
            size=13,
        )

        jan_ticks = np.arange(0, cluster_ts.shape[1], 12)
        ax.set_xticks(jan_ticks)
        ax.set_xticklabels(
            [f"Year {i}" for i in range(len(jan_ticks))],
            rotation=45,
            size=7,
            ha="right",
        )
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlabel("Time")

        map_ax.text(
            -0.05,
            0.95,
            "a)",
            transform=map_ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        ax.text(
            -0.1,
            1.09,
            f"{chr(98 + new_cluster_idx)})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        if new_cluster_idx == 0:
            patches_list = [
                mlines.Line2D([], [], color="black", label="Median Scenario"),
                mpatches.Patch(color=color, label="Q40–Q60", alpha=0.8),
                mpatches.Patch(color=color, label="Q30–Q70", alpha=0.6),
                mpatches.Patch(color=color, label="Q20–Q80", alpha=0.4),
                mpatches.Patch(color=color, label="Q10–Q90", alpha=0.2),
                mlines.Line2D(
                    [], [], color="#E34444", linestyle="--", label="Median Control"
                ),
            ]
            ax.legend(handles=patches_list, loc="best", fontsize=9)

    plt.savefig(
        os.path.join(save_path, f"combined_{n_clusters}_clusters_{title}.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":

    base_path = "./results/clustering"

    ds_47 = xr.open_dataset("./results/47/snow_47.nc")
    ds_16 = xr.open_dataset("./results/16/snow_16.nc")
    ds_150 = xr.open_dataset("./results/150/snow_150.nc")
    ds_ctrl = xr.open_dataset("./results/Control/snow_control.nc")

    datasets = {
        "47": ds_47,
        "16": ds_16,
        "150": ds_150,
        "Control": ds_ctrl,
        "all": [ds_ctrl, ds_16, ds_47, ds_150],
    }

    fraction_mask = xr.open_dataarray("./data/interim/land_mask_neu.nc")
    # Calculate cell area for weighting
    cell_area = compute_grid_cell_area(ds_47.snow_storage.isel(time=0))
    # Create land mask for extracting cell areas of only land cells
    land_mask = ~np.isnan(ds_47.snow_storage.isel(time=0))
    cell_area_1d = cell_area.values[land_mask.values]  # shape: (n_land_cells,)
    fractions_1d = fraction_mask.values[land_mask.values]  # shape: (n_land_cells,)

    timeseries_ctrl = prepare_time_series(ds_ctrl.snow_storage)
    timeseries_ctrl = timeseries_ctrl.squeeze()

    timeseries_scenario = prepare_time_series(ds_47.snow_storage)
    timeseries_scenario = timeseries_scenario.squeeze()

    labels = np.load("./results/clustering/47_Tg_dtw/5_cluster_labels.npy")

    plot_cluster_combined(
        da=ds_47.snow_storage,
        timeseries_scenario=timeseries_scenario,
        timeseries_ctrl=timeseries_ctrl,
        labels=labels,
        cell_areas=cell_area_1d,
        fractions=fractions_1d,
        n_clusters=5,
        title=None,
        parameter_name="Snow Storage (mm)",
        save_path="./results/clustering",
    )
