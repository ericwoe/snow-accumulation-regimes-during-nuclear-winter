import os

import cartopy.feature as cfeature
import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator, MultipleLocator

from src.utilities import (
    add_lat_lon_gridlines,
    change_time,
    compute_grid_cell_area,
)

DEFAULT_SCENARIO_COLORS = ["#abd9e9", "#74add1", "#4575b4"]
SOOT_INJECTION_MONTH = 4


def plot_combined_snow_analysis(
    *datasets,
    control=None,
    cell_area=None,
    mask=None,
    variable=None,
    labels=None,
    colors=None,
    output_path="./results/intercomparison/global_analysis.png",
):
    """
    Plot mean snow storage, snow storage anomaly, and snow cover extent
    anomaly for multiple scenarios relative to a control run.

    For each scenario dataset, this function computes:
      (a) the spatially weighted mean of `variable` over time,
      (b) its anomaly relative to the control run's monthly climatology mean
          (i.e., the control's mean for the corresponding calendar month),
      (c) the percentage of the weighted land area with snow storage > 0,
          expressed as an anomaly relative to the control's monthly
          climatology mean of snow cover extent.

    Results are plotted as three stacked panels sharing a common time axis. The
    control run's raw time series is shown for reference in the top panel; the
    control's monthly climatology mean and ±1 standard deviation are shown for
    reference in the two anomaly panels (b and c). A vertical dashed line marks
    the month of the soot injection.

    Parameters
    ----------
    *datasets : xr.Dataset
        One or more scenario datasets, each containing `variable` with
        dimensions ("time", "lat", "lon"). All datasets must share the same
        spatial grid and the same number of time steps.
    control : xr.Dataset
        Control run dataset, structured like the scenario datasets.
    cell_area : xr.DataArray
        Grid cell area [m2] with dimensions ("lat", "lon"), e.g. as returned
        by `compute_grid_cell_area`.
    mask : xr.DataArray
        Land fraction with dimensions ("lat", "lon"), ranging continuously
        from 0 (fully ocean) to 1 (fully land), used together with
        `cell_area` to weight the spatial average by the land area within
        each grid cell.
    variable : str
        Name of the variable in `datasets` and `control` to analyze (e.g.,
        "snow_storage").
    labels : list, optional
        Labels identifying each scenario (e.g., soot injection amounts in
        Tg), used in the legend. Must have the same length and order as
        `datasets`. Defaults to [16, 47, 150] if not provided.
    colors : list, optional
        Line colors for each scenario, matched by position to `datasets`.
        Defaults to `DEFAULT_SCENARIO_COLORS` if not provided.
    output_path : str, optional
        File path where the resulting figure is saved.

    Notes
    -----
    The anomaly calculations align each scenario time step with the
    control's climatological mean for the same calendar month
    (`groupby("time.month")`), rather than with a single time-mean control
    value. This accounts for the control run's own seasonal cycle.
    """

    if colors is None:
        colors = DEFAULT_SCENARIO_COLORS
    if labels is None:
        labels = [16, 47, 150]

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Combined area x land-mask weights used for all spatial averages below.
    # Restricted to cells with data (e.g. excluding cluster 0), so means and
    # snow cover fractions refer only to the analysed area, not all land.
    weights = (cell_area * mask).where(control[variable].isel(time=0).notnull())

    # --- Control run climatology ---

    mean_snow_control = (control[variable] * weights).sum(
        dim=["lat", "lon"]
    ) / weights.sum(dim=["lat", "lon"])
    # Compute the control run's monthly climatology mean and standard deviation
    mean_snow_control_monthly = mean_snow_control.groupby("time.month").mean()
    std_snow_control_monthly = mean_snow_control.groupby("time.month").std()

    # Snow cover extent for the control run: fraction of weighted land area
    # with snow_storage > 0, expressed as a percentage
    snow_control = control[variable] > 0
    snow_area_control = (snow_control * weights).sum(dim=("lat", "lon"))
    total_land_area_control = weights.sum(dim=("lat", "lon"))
    percentage_control = (snow_area_control / total_land_area_control) * 100

    # Compute the control run's monthly climatology mean of snow cover extent
    mean_snow_cover_control_monthly = percentage_control.groupby("time.month").mean()
    std_snow_cover_control_monthly = percentage_control.groupby("time.month").std()

    n_months = len(datasets[0][variable].time)
    month_index = np.arange(n_months)

    # --- Per-scenario calculations and plotting ---
    for ds, label, color in zip(datasets, labels, colors):
        # Select the variable of interest from the dataset
        da = ds[variable]

        # (a) Spatially weighted mean snow storage over time
        mean_snow_scenario = (da * weights).sum(dim=["lat", "lon"]) / weights.sum(
            dim=["lat", "lon"]
        )
        # Plot the mean snow storage for the scenario
        axes[0].plot(
            month_index, mean_snow_scenario.values, label=f"{label} Tg", color=color
        )

        # (b) Anomaly relative to the control's climatology for the same
        # calendar month
        anomaly_mean = (
            mean_snow_scenario.groupby("time.month") - mean_snow_control_monthly
        ).sortby("time")
        axes[1].plot(month_index, anomaly_mean.values, label=f"{label} Tg", color=color)

        # (c) Snow cover extent anomaly relative to the control's
        # climatology mean for the same calendar month
        snow = da > 0
        snow_area = (snow * weights).sum(dim=("lat", "lon"))
        total_land_area = weights.sum(dim=("lat", "lon"))
        percentage_scenario = (snow_area / total_land_area) * 100

        percentage_anomaly = (
            percentage_scenario.groupby("time.month") - mean_snow_cover_control_monthly
        ).sortby("time")
        # Plot the snow cover extent anomaly for the scenario
        axes[2].plot(
            month_index, percentage_anomaly.values, label=f"{label} Tg", color=color
        )

    # Add Control run reference lines and ±1σ shading for each panel
    time_index = datasets[0][variable].time
    std_expanded = std_snow_control_monthly.sel(month=time_index.dt.month).values
    std_expanded_cover = std_snow_cover_control_monthly.sel(
        month=time_index.dt.month
    ).values

    axes[0].plot(
        month_index,
        mean_snow_control.values,
        color="#E34444",
        linewidth=1.2,
        linestyle="--",
        label="Control",
    )

    axes[1].fill_between(
        month_index,
        -std_expanded,
        std_expanded,
        color="#CE5151",
        alpha=0.4,
        edgecolor="#CE5151",
        linewidth=1,
        label="±1σ Control",
    )

    axes[2].fill_between(
        month_index,
        -std_expanded_cover,
        std_expanded_cover,
        color="#CE5151",
        alpha=0.4,
        edgecolor="#CE5151",
        linewidth=1,
        label="±1σ Control",
    )

    # --- Axis formatting ---

    tick_positions = np.arange(0, n_months, 12)
    tick_labels = [f"Year {i}" for i in range(len(tick_positions))]

    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_xlim(-1, n_months)
        ax.grid(which="major", axis="x", linewidth=0.8, color="gray", alpha=0.5)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(which="minor", axis="x", linewidth=0.3, color="gray", alpha=0.3)
        ax.grid(which="major", axis="y", linewidth=0.5, color="gray", alpha=0.3)
        ax.legend(fontsize=9)

    axes[2].set_xlabel("Time [months]", fontsize=11)
    axes[0].set_ylabel("Mean Snow Storage [mm]", fontsize=12)
    # Add vertical dashed line marking the month of soot injection (May of Year 0)
    axes[0].axvline(
        SOOT_INJECTION_MONTH, color="black", linewidth=0.5, linestyle="dashed"
    )

    axes[1].set_ylabel("Mean Snow Storage Anomaly [mm]", fontsize=12)
    axes[1].axhline(0, color="#E34444", linewidth=0.8, linestyle="--")
    # Add vertical dashed line marking the month of soot injection (May of Year 0)
    axes[1].axvline(
        SOOT_INJECTION_MONTH, color="black", linewidth=0.5, linestyle="dashed"
    )

    axes[2].set_ylabel("Snow Cover Extent Anomaly [pp]", fontsize=12)
    axes[2].axhline(0, color="#E34444", linewidth=1.2, linestyle="--")
    # Add vertical dashed line marking the month of soot injection (May of Year 0)
    axes[2].axvline(
        SOOT_INJECTION_MONTH, color="black", linewidth=0.5, linestyle="dashed"
    )

    # Panel labels (a, b, c)
    for i, ax in enumerate(axes):
        ax.text(
            -0.05,
            1.02,
            f"{chr(97 + i)})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def find_mean_anomaly_peak_time(
    scenario: xr.Dataset,
    control: xr.Dataset,
    cell_area: xr.DataArray,
    mask: xr.DataArray,
    variable: str = "snow_storage",
):
    """
    Identify the time point of peak spatially weighted mean anomaly
    relative to the control run's monthly climatology, for a single
    scenario.

    The anomaly is computed as the spatially weighted average of
    `variable`, relative to the control run's monthly climatology mean
    (i.e., the control's mean for the corresponding calendar month). This
    removes normal seasonal and interannual variability in the control run
    before identifying the peak deviation.

    Parameters
    ----------
    scenario : xr.Dataset
        Scenario dataset containing `variable` with dimensions
        ("time", "lat", "lon").
    control : xr.Dataset
        Control run dataset, structured like `scenario` and sharing the
        same time coordinate values.
    cell_area : xr.DataArray
        Grid cell area [m2] with dimensions ("lat", "lon").
    mask : xr.DataArray
        Land fraction with dimensions ("lat", "lon"), ranging continuously
        from 0 (fully ocean) to 1 (fully land).
    variable : str, optional
        Name of the variable in `scenario` and `control` to analyze.
        Defaults to "snow_storage".

    Returns
    -------
    time_mean_peak : cftime
        Time value at which the spatially weighted mean climatological
        anomaly reaches its maximum.

    Notes
    -----
    Both `scenario` and `control` should already be restricted to the
    spatial domain of interest (e.g., a specific cluster), since this
    function does not perform any cluster filtering itself.
    """
    # Restricted to cells with data (e.g. excluding cluster 0), as in
    # plot_combined_snow_analysis
    weights = (cell_area * mask).where(control[variable].isel(time=0).notnull())

    mean_snow_control = (control[variable] * weights).sum(
        dim=["lat", "lon"]
    ) / weights.sum(dim=["lat", "lon"])
    mean_snow_control_monthly = mean_snow_control.groupby("time.month").mean()

    mean_snow_scenario = (scenario[variable] * weights).sum(
        dim=["lat", "lon"]
    ) / weights.sum(dim=["lat", "lon"])

    anomaly_mean = (
        mean_snow_scenario.groupby("time.month") - mean_snow_control_monthly
    ).sortby("time")

    time_mean_peak = anomaly_mean.idxmax(dim="time").item()

    return time_mean_peak


def plot_snow_anomaly_spatial(
    scenario: xr.Dataset,
    control: xr.Dataset,
    time_point: cftime,
    scenario_name: str,
    variable: str = "snow_storage",
    output_path: str = "./results/intercomparison/timepoint_max_snow_anomaly_spatial.png",
    vmin: float = -5,
    vmax: float = 5,
    linthresh: float = 0.1,
    sim_start_year: int = None,  # falls None: wird aus den Daten abgeleitet
):
    """
    Plot the spatial snow storage anomaly.
    Uses symmetric log scaling to emphasize the dominant range [-linthresh, linthresh] m.
    Title shows the calendar month and the simulation year (relative to sim_start_year)
    instead of the absolute calendar date.
    """

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    cmap = plt.get_cmap("RdBu")
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.Robinson()})

    # Calculate anomaly
    anomaly = (
        scenario[variable].sel(time=time_point) - control[variable].sel(time=time_point)
    ) / 1000

    norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

    ax.add_feature(cfeature.OCEAN, color="lightgrey", zorder=10)
    ax.coastlines(linewidth=0.5, zorder=11)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=11)

    add_lat_lon_gridlines(ax, linewidth=0.8, fontsize=10)

    im = anomaly.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
    )

    # Determine simulation start year from data if not explicitly provided
    if sim_start_year is None:
        sim_start_year = scenario["time"].min().item().year

    sim_year = time_point.year - sim_start_year
    month_name = month_names[time_point.month - 1]

    ax.set_title(
        f"{scenario_name} – {month_name}, Year {sim_year}",
        fontsize=12,
    )

    cbar = fig.colorbar(
        im, ax=ax, orientation="vertical", fraction=0.03, pad=0.04, extend="both"
    )
    cbar.set_label("Snow Storage Anomaly [m]", fontsize=11)

    ticks = sorted(set([vmin, -2, -1, -0.5, -linthresh, 0, linthresh, 0.5, 1, 2, vmax]))
    ticks = [t for t in ticks if vmin <= t <= vmax]
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_minor_locator(FixedLocator([]))
    cbar.set_ticklabels([f"{t:g}" for t in ticks])

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hovmoeller_mean_snow_storage(
    *datasets,
    control,
    mask=None,
    cell_area=None,
    savedir="./results/allgemeine_muster",
    vmin=-5,
    vmax=5,
    linthresh=0.1,
    titles=None,
):
    """
    Erstellt Hovmöller-Diagramme der zonal gemittelten snow_storage Anomalie
    (Szenario - Control, zeitlich 1:1 gleicher Index) für mehrere Szenarien
    in einem Multi-Panel-Plot.
    """
    os.makedirs(savedir, exist_ok=True)
    n = len(datasets)

    if mask is not None and cell_area is not None:
        weights = mask * cell_area
    elif cell_area is not None:
        weights = cell_area
    elif mask is not None:
        weights = mask
    else:
        weights = None

    def zonal_mean(da):
        if weights is not None:
            weights_masked = weights.where(da.notnull())
            weighted_sum = (da * weights_masked).sum(dim="lon")
            total_weight = weights_masked.sum(dim="lon")
            return weighted_sum / total_weight.where(total_weight > 0)
        return da.mean(dim="lon")

    zonal_control = zonal_mean(control.snow_storage)

    all_zonal = []
    for ds in datasets:
        zonal_scenario = zonal_mean(ds.snow_storage)
        zonal_anomaly = (zonal_scenario.values - zonal_control.values) / 1000  # mm -> m
        zonal_anomaly = xr.DataArray(
            zonal_anomaly,
            dims=zonal_scenario.dims,
            coords=zonal_scenario.coords,
        )
        all_zonal.append(zonal_anomaly)

    n_months = len(datasets[0].time)
    tick_positions = np.arange(0, n_months, 12)
    tick_labels = [f"Year {i}" for i in range(len(tick_positions))]

    fig, axes = plt.subplots(
        n, 1, figsize=(14, 3 * n), sharex=True, sharey=True, constrained_layout=True
    )
    if n == 1:
        axes = [axes]
    # Reserve headroom so the top panel's label doesn't get clipped at the
    # figure edge; constrained_layout doesn't account for it otherwise.
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.96))

    norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdBu")

    for i, (ax, ds, zonal) in enumerate(zip(axes, datasets, all_zonal)):
        time_vals = np.arange(len(ds.time))
        lat_vals = ds.lat.values
        plot_data = np.nan_to_num(zonal.values.T, nan=0.0)

        im = ax.pcolormesh(
            time_vals,
            lat_vals,
            plot_data,
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

        ax.set_ylabel("Latitude [°]")
        label = titles[i] if titles and i < len(titles) else ds.case
        ax.set_title(f"{label} Tg", fontsize=11)

        # Offset above the axes far enough to clear the title set above,
        # anchored at a fixed point distance rather than a fractional
        # y-position so it can't collide with a tick label at the top edge.
        ax.annotate(
            f"{chr(97 + i)})",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 14),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis="x", direction="out", length=4, width=0.8, bottom=True)
        ax.tick_params(axis="y", direction="out", length=4, width=0.8, left=True)

    axes[-1].set_xlabel("Time")

    cbar = fig.colorbar(
        im,
        ax=axes,
        label="Zonal Mean Snow Storage Anomaly [m]",
        location="right",
        shrink=0.6,
        pad=0.015,
        aspect=30,
        extend="both",
    )

    ticks = sorted(set([vmin, -2, -1, -0.5, -linthresh, 0, linthresh, 0.5, 1, 2, vmax]))
    ticks = [t for t in ticks if vmin <= t <= vmax]
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_minor_locator(FixedLocator([]))
    cbar.set_ticklabels([f"{t:g}" for t in ticks])

    fig.savefig(
        os.path.join(savedir, "hovmoeller_mean_snow_storage_anomaly.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    ds_16 = xr.open_dataset("./results/16/snow_16.nc")
    ds_47 = xr.open_dataset("./results/47/snow_47.nc")
    ds_150 = xr.open_dataset("./results/150/snow_150.nc")
    ds_ctrl = xr.open_dataset("./results/Control/snow_control.nc")
    mask = xr.open_dataarray("./data/interim/land_mask_neu.nc")

    cell_area = compute_grid_cell_area(ds_ctrl.snow_storage.isel(time=0))

    # Change the time coordinate of each dataset to align with the intended month labeling
    for ds in [ds_16, ds_47, ds_150, ds_ctrl]:
        change_time(ds)

    # Exclude Cluster 0 (permanently accumulating cells from control run) from analysis
    # load cluster labels
    labels = np.load("./results/clustering/Control_scenario_dtw/3_cluster_labels.npy")
    # create a cluster map with the same shape as the snow_storage variable
    template = ds_ctrl.snow_storage.isel(time=0)
    cluster_map = xr.full_like(template, fill_value=np.nan)
    land_mask = ~np.isnan(ds_ctrl.snow_storage.isel(time=0))
    cluster_map.values[land_mask.values] = labels
    # Select only the grid cells that belong to clusters 1 and 2 (exclude cluster 0)
    ds_47_0 = ds_47.where(cluster_map != 0)
    ds_16_0 = ds_16.where(cluster_map != 0)
    ds_150_0 = ds_150.where(cluster_map != 0)
    ds_ctrl_0 = ds_ctrl.where(cluster_map != 0)

    plot_combined_snow_analysis(
        ds_16_0,
        ds_47_0,
        ds_150_0,
        control=ds_ctrl_0,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
        labels=[16, 47, 150],
        colors=DEFAULT_SCENARIO_COLORS,
    )

    time_mean_peak = find_mean_anomaly_peak_time(
        scenario=ds_47_0,
        control=ds_ctrl_0,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )

    plot_snow_anomaly_spatial(
        scenario=ds_47,
        control=ds_ctrl,
        time_point=time_mean_peak,
        scenario_name="47 Tg",
        variable="snow_storage",
        output_path="./results/intercomparison/timepoint_max_snow_anomaly_spatial.png",
        vmin=-5,
        vmax=5,
        linthresh=0.1,
    )

    plot_hovmoeller_mean_snow_storage(
        ds_16,
        ds_47,
        ds_150,
        control=ds_ctrl,
        mask=mask,
        cell_area=cell_area,
        savedir="./results/intercomparison/hovmoeller/mean_snow_storage",
        titles=[16, 47, 150],
        vmin=-5,
        vmax=5,
        linthresh=0.1,
    )
