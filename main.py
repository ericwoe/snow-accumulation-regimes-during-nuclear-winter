"""
End-to-end pipeline for the snow storage / snowmelt nuclear-winter analysis.

Given CESM output for a control run and three soot-injection scenarios
(16, 47, 150 Tg), this script:

1. Preprocessing: downloads the raw nuclear winter climate model output if needed, derives
   precipitation and temperature variables, and builds the land mask (by
   downloading a Natural Earth land-polygon shapefile) or loads it from
   cache if already built.
2. Snow model: runs the snow accumulation/melt model on each scenario
   (prepended with a spin-up period from the control run), restricts the
   result to the analysis period, and caches it to ``results/<scenario>/``.
3. Clustering: k-means clustering of grid-cell snow storage time series,
   used to flag a cluster of permanently snow/ice-accumulating cells
   (Cluster 0, e.g. Antarctica/Greenland) that is excluded from the
   anomaly analyses below. DTW k-means on the full grid is expensive
   (minutes to hours), so this step caches its labels/model to
   ``results/<scenario>/clustering/models/`` and skips recomputation on
   later runs.
4. Global snow analysis: control-vs-scenario comparisons of snow storage
   (combined summary plot, spatial anomaly map at the peak anomaly time,
   Hovmoeller diagrams) across all three scenarios.
5. Basin analysis: downloads (and caches) HydroBasins catchment shapefiles
   for six major river basins, then converts snowmelt and rain to discharge
   (m^3/s) and produces annual-anomaly, monthly, and seasonal-cycle figures
   comparing the 47 Tg scenario to the control run.

All figures and intermediate datasets are written under ``results/``.
"""

import pickle
from pathlib import Path

import cftime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.postprocessing.basin_analysis import (
    convert_mm_month_to_discharge_m3_month,
    monthly_discharge_sum_m3,
    plot_annual_anomaly,
    plot_monthly_bars_rain_snow,
    plot_seasonal_cycle_phases,
)
from src.postprocessing.plot_clustering import (
    plot_cluster_combined,
    plot_elbow,
)
from src.postprocessing.snow_analysis import (
    DEFAULT_SCENARIO_COLORS,
    find_mean_anomaly_peak_time,
    plot_combined_snow_analysis,
    plot_hovmoeller_mean_snow_storage,
    plot_snow_anomaly_spatial,
)
from src.preprocessing import prepare_data
from src.preprocessing.download_data import download_model_output, load_basin_region
from src.processing.clustering import (
    elbow_method,
    prepare_time_series,
    time_series_analysis,
)
from src.processing.snow_model import add_snow_variables
from src.utilities import change_time, compute_grid_cell_area, create_mask

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)

SCENARIOS = {
    "Control": {
        "data_dir": "./data/cesm_output_from_harrison/temp_precip/nw_cntrl_03",
        "pattern": "*.nc",
        "result_path": "./results/Control/snow_model_output_control.nc",
        "osf_url": "https://osf.io/download/w6qb5/",
        "osf_archive": "nw_cntrl_03.TS_TSMN_TSMX_PRECC_PRECL_v2024-12-12.nc.tar.gz",
    },
    "16": {
        "data_dir": "./data/cesm_output_from_harrison/temp_precip/nw_targets_04",
        "pattern": "*.nc",
        "result_path": "./results/16/snow_model_output_16Tg.nc",
        "osf_url": "https://osf.io/download/n2vtj/",
        "osf_archive": "nw_targets_04.TS_TSMN_TSMX_PRECC_PRECL_v2024-12-12.nc.tar.gz",
    },
    "47": {
        "data_dir": "./data/cesm_output_from_harrison/temp_precip/nw_targets_05",
        "pattern": "*.nc",
        "result_path": "./results/47/snow_model_output_47Tg.nc",
        "osf_url": "https://osf.io/download/4uysj/",
        "osf_archive": "nw_targets_05.TS_TSMN_TSMX_PRECC_PRECL_v2024-12-12.nc.tar.gz",
    },
    "150": {
        "data_dir": "./data/cesm_output_from_harrison/temp_precip/nw_ur_150_07",
        "pattern": "*.nc",
        "result_path": "./results/150/snow_model_output_150Tg.nc",
        "osf_url": "https://osf.io/download/wnvxf/",
        "osf_archive": "nw_ur_150_07.TS_TSMN_TSMX_PRECC_PRECL_v2024-12-12.nc.tar.gz",
    },
}

RESULT_DIRS = {
    name: Path(config["result_path"]).parent for name, config in SCENARIOS.items()
}

LAND_SHAPE_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
LAND_MASK_FILE_PATH = "./data/land_fraction_mask/land_mask_neu.nc"
# Start/end are deliberately set to month 2 and month 1 (not month 1 for both):
# change_time() later shifts every timestamp back by one month (raw timestamps
# mark the end of the averaging period). This offset exactly compensates for
# that, so after slicing + shifting we get full calendar years with no leftover
# single-month year at the boundary. If either month is changed to 1 without
# adjusting the other accordingly, you'll end up with a year that has only one
# month.
SPINUP_START = cftime.DatetimeNoLeap(1, 2, 1, 0, 0, 0, 0, has_year_zero=True)
SPINUP_END = cftime.DatetimeNoLeap(5, 1, 1, 0, 0, 0, 0, has_year_zero=True)
ANALYSIS_START = cftime.DatetimeNoLeap(5, 2, 1, 0, 0, 0, 0, has_year_zero=True)
ANALYSIS_END = cftime.DatetimeNoLeap(20, 1, 1, 0, 0, 0, 0, has_year_zero=True)


datasets = {}

# Read in Data and calculate precip and temperature variables
print("\n=== PREPROCESSING ===")
print("Checking model output data...")
for name, config in SCENARIOS.items():
    download_model_output(name, config)
for name, config in SCENARIOS.items():
    ds = prepare_data.run_preprocessing(
        data_directory=config["data_dir"],
        file_pattern=config["pattern"],
        output_path=None,
    )
    datasets[name] = ds

# Extract spinup period from Control dataset
spinup_period = datasets["Control"].sel(time=slice(SPINUP_START, SPINUP_END))

# Create or import land mask (same grid for all scenarios)
land_mask_path = Path(LAND_MASK_FILE_PATH)
if land_mask_path.exists():
    print("Loading existing land mask")
    mask = xr.open_dataarray(land_mask_path)
else:
    print("Creating land mask")
    gadm = gpd.read_file(LAND_SHAPE_URL)
    mask = create_mask(datasets["Control"], gadm)
    print(f"Saving Mask to {LAND_MASK_FILE_PATH}")
    land_mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.to_netcdf(land_mask_path)


for name, config in SCENARIOS.items():
    result_path = Path(config["result_path"])

    if result_path.exists():
        print(f"Loading existing {name} dataset with snow variables from {result_path}")
        ds = xr.open_dataset(result_path)
    else:
        ds = datasets[name]

        # Prepend spinup to non-Control datasets so the snow model initializes correctly
        if name != "Control":
            ds = xr.concat([spinup_period, ds], dim="time")

        # Run Snow Model on the full time series (spinup included) for correct initialization
        ds = add_snow_variables(ds)

        # Reduce to Analysis Period
        ds = ds.sel(time=slice(ANALYSIS_START, ANALYSIS_END))

        # Apply land mask - values become NaN where mask == 0
        print("Applying land mask to dataset")
        ds = ds.where(mask > 0)

        # Delete time_bnds to avoid saving conflict
        ds = ds.drop_vars("time_bnds")

        # Save preprocessed dataset
        print(f"Saving {name} dataset to {result_path}")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(result_path)

    datasets[name] = ds

# compute cell area for area weighted means
cell_area = compute_grid_cell_area(datasets["Control"].snow_storage.isel(time=0))

# Shift time coordinates back by one month for all datasets
for ds in datasets.values():
    change_time(ds)


##################################################################################################
# CLUSTERING - 47 Tg Scenario and Control
##################################################################################################

print("\n=== CLUSTERING ===")
# Goal: group grid cells by the *shape* of their snow_storage time series
# (via DTW k-means), so that the Global Snow Analysis section further below
# can exclude the cluster of permanently snow/ice-covered cells (e.g.
# Antarctica, Greenland) from its area-weighted means - those cells would
# otherwise dominate the average and mask the seasonal signal elsewhere.
# Only Control and the 47 Tg scenario are clustered: Control to find that
# "permanent snow" cluster in the baseline climate, and 47 Tg to visually
# check (via plot_cluster_combined below) that soot injection doesn't
# fundamentally reshape which cells belong to it.
for name, ds in datasets.items():
    if name not in ["47", "Control"]:
        continue

    print(f"\n--- Clustering: {name} ---")
    clustering_dir = RESULT_DIRS[name] / "clustering"
    models_dir = clustering_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Cluster counts were chosen per scenario beforehand (via the elbow plots
    # saved below) and are not recomputed automatically here.
    n_clusters = 5 if name == "47" else 3
    labels_path = models_dir / f"{n_clusters}_cluster_labels.npy"
    model_path = models_dir / f"kmeans_model_{n_clusters}_clusters.pkl"

    # prepare_time_series flattens the (time, lat, lon) grid to land cells
    # only, shape (n_land_cells, time_steps, 1) - the format tslearn expects.
    print(f"  {name}: extracting land-cell time series from snow_storage")
    timeseries = prepare_time_series(ds.snow_storage)
    print(f"  {name}: {timeseries.shape[0]} land cells, {timeseries.shape[1]} months")

    # --- Step 1: elbow method, always run ---
    # Diagnostic (inertia vs. k) to help pick n_clusters by eye; it doesn't
    # feed back into n_clusters above. Runs on a cheaper random 30% subset
    # of the land cells rather than the full grid. elbow_method() itself
    # checkpoints per k in save_dir, so re-running here is cheap once all
    # k values are cached - it just reloads them.
    subset_size = int(0.3 * timeseries.shape[0])
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(timeseries.shape[0], subset_size, replace=False)
    timeseries_subset = timeseries[indices]
    print(
        f"  {name}: running elbow method (k=2..10) on a {subset_size}-cell "
        "subset for the diagnostic plot"
    )
    inertias_df = elbow_method(
        timeseries_subset, max_clusters=10, save_dir=clustering_dir / "elbow"
    )
    plot_elbow(inertias_df, save_dir=clustering_dir / "elbow")
    print(f"  {name}: elbow plot saved to {clustering_dir / 'elbow'}")

    # --- Step 2: full clustering at the chosen n_clusters, cached ---
    # DTW k-means on the full grid is expensive (minutes to hours), so skip
    # it if a previous run already produced and saved this scenario's labels.
    if labels_path.exists():
        print(
            f"  {name}: loading existing {n_clusters}-cluster result from {labels_path}"
        )
        labels = np.load(labels_path)
    else:
        print(
            f"  {name}: no cached result, running full {n_clusters}-cluster DTW "
            f"k-means on all {timeseries.shape[0]} land cells"
        )
        labels, km = time_series_analysis(timeseries, n_clusters=n_clusters)

        # Save cluster labels and model so future runs can skip this step
        np.save(labels_path, labels)
        with open(model_path, "wb") as f:
            pickle.dump(km, f)
        print(
            f"  {name}: saved cluster labels to {labels_path} and model to {model_path}"
        )

    if name == "47":
        # Diagnostic-only figure (spatial cluster map + per-cluster time
        # series for scenario vs. control). Its clusters are renumbered by
        # ascending mean snow storage, so its IDs do not match the raw Control
        # labels used for `cluster_map != 0` in the Global Snow Analysis
        # section below; it doesn't feed back into the labels themselves.
        print(f"  {name}: building scenario-vs-control cluster comparison plot")
        timeseries_control = prepare_time_series(datasets["Control"].snow_storage)

        # Flatten cell_area/mask the same way prepare_time_series flattened
        # snow_storage (land cells only, same iteration order), so cell_area_1d
        # and fractions_1d line up index-for-index with timeseries/labels.
        land_mask = ~np.isnan(ds.snow_storage.isel(time=0))
        cell_area_1d = cell_area.values[land_mask.values]
        fractions_1d = mask.values[land_mask.values]

        timeseries_squeezed = timeseries.squeeze()
        timeseries_control_squeezed = timeseries_control.squeeze()

        plot_cluster_combined(
            da=ds.snow_storage,
            timeseries_scenario=timeseries_squeezed,
            timeseries_ctrl=timeseries_control_squeezed,
            labels=labels,
            cell_areas=cell_area_1d,
            fractions=fractions_1d,
            n_clusters=n_clusters,
            parameter_name="Snow Storage [mm]",
            save_path=clustering_dir,
        )
        print(f"  {name}: cluster comparison plot saved to {clustering_dir}")


##################################################################################################
# POSTPROCESSING - Global Snow Analysis
##################################################################################################

print("\n=== GLOBAL SNOW ANALYSIS ===")

ds_47 = datasets["47"]
ds_control = datasets["Control"]

global_analysis_dir = RESULT_DIRS["47"] / "global_analysis"
global_analysis_dir.mkdir(parents=True, exist_ok=True)

# Exclude Cluster 0 (permanently accumulating cells) from analysis
# Load cluster labels for Control dataset
print("Loading Control cluster labels to identify permanent-snow cells")
labels_ctrl = np.load(
    RESULT_DIRS["Control"] / "clustering" / "models" / "3_cluster_labels.npy"
)
# Map cluster labels to the spatial grid of the Control dataset
template = ds_control.snow_storage.isel(time=0)
cluster_map = xr.full_like(template, fill_value=np.nan)
land_mask_ctrl = ~np.isnan(ds_control.snow_storage.isel(time=0))
cluster_map.values[land_mask_ctrl.values] = labels_ctrl

# Exclude Cluster 0 from all datasets for analysis
ds_47_0 = ds_47.where(cluster_map != 0)
ds_16_0 = datasets["16"].where(cluster_map != 0)
ds_150_0 = datasets["150"].where(cluster_map != 0)
ds_ctrl_0 = ds_control.where(cluster_map != 0)

print("Plotting combined snow analysis (16/47/150 Tg vs. control)")
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
    output_path=global_analysis_dir / "global_analysis.png",
)
print(f"  saved to {global_analysis_dir / 'global_analysis.png'}")


# Find the calendar month/year where the 47 Tg scenario's spatially
# weighted mean snow-storage anomaly (relative to the control's monthly
# climatology) is largest. Cluster 0 is excluded here so permanently
# snow/ice-covered cells don't dominate the area-weighted mean.
print("Finding peak snow-storage anomaly time for the 47 Tg scenario")
time_mean_peak = find_mean_anomaly_peak_time(
    scenario=ds_47_0,
    control=ds_ctrl_0,
    cell_area=cell_area,
    mask=mask,
    variable="snow_storage",
)
print(f"  peak anomaly at {time_mean_peak}")

# Map of the spatial anomaly pattern at that peak time. Uses the full
# (cluster-0-included) datasets, unlike the analyses above, so the map
# itself isn't missing those cells.
print("Plotting spatial snow anomaly map at the peak time")
plot_snow_anomaly_spatial(
    scenario=ds_47,
    control=ds_control,
    time_point=time_mean_peak,
    scenario_name="47 Tg",
    variable="snow_storage",
    output_path=global_analysis_dir / "timepoint_max_snow_anomaly_spatial.png",
    vmin=-5,
    vmax=5,
    linthresh=0.1,
)
print(f"  saved to {global_analysis_dir / 'timepoint_max_snow_anomaly_spatial.png'}")

# Hovmoeller diagrams (zonal-mean anomaly vs. time) for all three
# scenarios against the control run; also uses the full datasets.
print("Plotting Hovmoeller diagrams (16/47/150 Tg vs. control)")
plot_hovmoeller_mean_snow_storage(
    datasets["16"],
    ds_47,
    datasets["150"],
    control=ds_control,
    mask=mask,
    cell_area=cell_area,
    savedir=global_analysis_dir,
    titles=[16, 47, 150],
    vmin=-5,
    vmax=5,
    linthresh=0.1,
)
print(f"  saved to {global_analysis_dir / 'hovmoeller_mean_snow_storage_anomaly.png'}")

##################################################################################################
# POSTPROCESSING - Basin Analysis
##################################################################################################

print("\n=== BASIN ANALYSIS ===")

_HYDROBASINS_BASE = "https://data.hydrosheds.org/file/hydrobasins/standard"
HYDROBASINS_URLS = {
    "eu": f"{_HYDROBASINS_BASE}/hybas_eu_lev01-12_v1c.zip",
    "as": f"{_HYDROBASINS_BASE}/hybas_as_lev01-12_v1c.zip",
    "si": f"{_HYDROBASINS_BASE}/hybas_si_lev01-12_v1c.zip",
    "na": f"{_HYDROBASINS_BASE}/hybas_na_lev01-12_v1c.zip",
}
HYDROBASINS_DIR = Path("./data/hydrobasins_level4")

MAIN_RIVERS = {
    "Rhine": ("eu", 2040023010),
    "Ganges-Brahmaputra-Meghna": ("as", 4040025450),
    "Yellow River": ("as", 4040007850),
    "Tigris-Euphrates": ("eu", 2040073570),
    "Ob": ("si", 3040001840),
    "Mississippi": ("na", 7040047060),
}

basin_analysis_dir = RESULT_DIRS["47"] / "basin_analysis"
basin_analysis_dir.mkdir(parents=True, exist_ok=True)

BASIN_VARIABLES = {
    "snowmelt": "snow_melt",
    "rain": "rain",
}
scenario_datasets = {"47": ds_47, "control": ds_control}

# For each river basin below: load its polygon, build a fractional basin
# mask on the model grid, then compute discharge from snowmelt and rain for
# both the 47 Tg scenario and the control run. The three dicts collected
# here each feed a different downstream figure:
#   - annual_total_discharge_anomaly_data, annual_snowmelt_anomaly_data:
#     annual-anomaly bar charts (plot_annual_anomaly)
#   - monthly_snow_rain_data: monthly-bar and seasonal-cycle discharge-rate
#     plots (plot_monthly_bars_rain_snow, plot_seasonal_cycle_phases)
annual_snowmelt_anomaly_data = {}
annual_total_discharge_anomaly_data = {}
monthly_snow_rain_data = {}

# In-memory cache to avoid re-reading within one run (Rhine and Tigris-Euphrates
# share HydroBasins region "eu", Ganges-Brahmaputra-Meghna and Yellow River
# share "as")
_basin_gdf_cache = {}

for river, (region, river_id) in MAIN_RIVERS.items():
    print(f"  {river}: computing basin discharge")
    # Load (and cache) the HydroBasins region, then pick out and merge all
    # level-4 sub-basins that belong to this river's main basin ID.
    if region not in _basin_gdf_cache:
        _basin_gdf_cache[region] = load_basin_region(
            region, HYDROBASINS_DIR, HYDROBASINS_URLS
        )
    basins_lev4 = _basin_gdf_cache[region]
    river_basin = basins_lev4[basins_lev4["MAIN_BAS"] == river_id]
    river_basin_diss = river_basin.dissolve()

    # river_mask is the fraction (0-1) of each grid cell covered by the
    # basin polygon - same convention as the global land `mask` above, just
    # restricted to this one basin's outline.
    river_mask = create_mask(ds_control, river_basin_diss)

    dsc_annual = {}
    dsc_m3_s = {}
    for var_key, var_name in BASIN_VARIABLES.items():
        for scen_key, scen_ds in scenario_datasets.items():
            # Drop cells with zero overlap with the basin (NaN, so they're
            # excluded from the later sum), then convert mm/month to
            # m^3/month - this is where cells partially inside the basin
            # get weighted down by river_mask.
            masked = scen_ds[var_name].where(river_mask > 0)
            discharge_m3 = convert_mm_month_to_discharge_m3_month(
                masked, river_mask, cell_area
            )
            # Sum over the grid to get one basin-total time series.
            dsc_sum = monthly_discharge_sum_m3(discharge_m3)

            # Annual totals, for the annual-anomaly bar charts.
            dsc_annual[scen_key, var_key] = dsc_sum.groupby("time.year").sum("time")

            # Mean discharge rate in m^3/s, for the monthly/seasonal plots.
            # days_in_month is a calendar quantity, not a spatial one;
            # derive it straight from the (already correctly shifted) time
            # axis instead of reading it off an arbitrary grid cell.
            seconds_in_month = scen_ds.time.dt.days_in_month * 24 * 3600
            dsc_m3_s[scen_key, var_key] = dsc_sum / seconds_in_month

    annual_total_discharge_anomaly_data[river] = (
        dsc_annual["47", "snowmelt"] + dsc_annual["47", "rain"],
        dsc_annual["control", "snowmelt"] + dsc_annual["control", "rain"],
    )

    annual_snowmelt_anomaly_data[river] = (
        dsc_annual["47", "snowmelt"],
        dsc_annual["control", "snowmelt"],
    )

    monthly_snow_rain_data[river] = {
        "scenario_snowmelt": dsc_m3_s["47", "snowmelt"],
        "scenario_rain": dsc_m3_s["47", "rain"],
        "control_snowmelt": dsc_m3_s["control", "snowmelt"],
        "control_rain": dsc_m3_s["control", "rain"],
    }

print(f"Done computing discharge for all {len(MAIN_RIVERS)} river basins")

# --- Multi-panel figures ---

print("Plotting annual total discharge anomalies")
# Annual Anomalies of Total Discharge (Rain + Snowmelt)
n_rivers = len(annual_total_discharge_anomaly_data)
ncols = 3
nrows = (n_rivers + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows, ncols, figsize=(ncols * 6, nrows * 5), constrained_layout=True
)
# Reserve a sliver of headroom at the top of the figure: constrained_layout
# doesn't account for the panel-label annotations below when sizing the
# axes grid, so without this the top row's label boxes get clipped.
fig.get_layout_engine().set(rect=(0, 0, 1, 0.96))
axes_flat1 = axes.flatten()

for idx, (ax, (river, (scenario, control))) in enumerate(
    zip(axes_flat1, annual_total_discharge_anomaly_data.items())
):
    plot_annual_anomaly(
        scenario, control, river=river, ax=ax, title="Annual Total Discharge"
    )
    # Anchored a fixed 6pt above the axes' top-left corner rather than at a
    # fractional (data-dependent) y-position, so the label can't collide
    # with a y-tick label that happens to sit right at the top of the axis.
    ax.annotate(
        f"{chr(97 + idx)})",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=12,
        weight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

y_max = max(ax.get_ylim()[1] for ax in axes_flat1[:n_rivers])
y_min = min(ax.get_ylim()[0] for ax in axes_flat1[:n_rivers])
for ax in axes_flat1[:n_rivers]:
    ax.set_ylim(y_min, y_max, auto=False)
for ax in axes_flat1[n_rivers:]:
    ax.set_visible(False)

plt.savefig(basin_analysis_dir / "river_basin_annual_anomalies.png", dpi=600)
plt.close(fig)
print(f"  saved to {basin_analysis_dir / 'river_basin_annual_anomalies.png'}")

print("Plotting annual snowmelt anomalies")
# Annual Anomalies of Snowmelt
fig4, axes4 = plt.subplots(
    nrows, ncols, figsize=(ncols * 6, nrows * 5), constrained_layout=True
)
fig4.get_layout_engine().set(rect=(0, 0, 1, 0.96))
axes_flat4 = axes4.flatten()

for idx, (ax, (river, (scenario, control))) in enumerate(
    zip(axes_flat4, annual_snowmelt_anomaly_data.items())
):
    plot_annual_anomaly(scenario, control, river=river, ax=ax, title="Snowmelt")
    ax.annotate(
        f"{chr(97 + idx)})",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=12,
        weight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

for ax in axes_flat4[n_rivers:]:
    ax.set_visible(False)

plt.savefig(basin_analysis_dir / "river_basin_annual_snowmelt_anomalies.png", dpi=600)
plt.close(fig4)
print(f"  saved to {basin_analysis_dir / 'river_basin_annual_snowmelt_anomalies.png'}")

print("Plotting monthly discharge bars (all rivers)")
# Monthly Mean Discharge from Snowmelt and Rain
# Own grid shape (2 columns, taller) rather than the nrows/ncols used above -
# derived from n_rivers so a changed MAIN_RIVERS count doesn't silently drop
# rivers out of the zip() below.
ncols_monthly = 2
nrows_monthly = (n_rivers + ncols_monthly - 1) // ncols_monthly
fig3, axes3 = plt.subplots(
    nrows_monthly,
    ncols_monthly,
    figsize=(ncols_monthly * 6, nrows_monthly * 5),
    constrained_layout=True,
)
fig3.get_layout_engine().set(rect=(0, 0, 1, 0.96))
axes_flat3 = axes3.flatten()

for idx, (ax, (river, river_data)) in enumerate(
    zip(axes_flat3, monthly_snow_rain_data.items())
):
    plot_monthly_bars_rain_snow(
        river_data["scenario_snowmelt"],
        river_data["scenario_rain"],
        river_data["control_snowmelt"],
        river_data["control_rain"],
        river,
        ax=ax,
    )
    # Offset further above the axes than in the other figures: this plot
    # also carries a centered river-name title (set inside
    # plot_monthly_bars_rain_snow), so the panel label needs to clear that
    # title's height too, not just the axes top edge.
    ax.annotate(
        f"{chr(97 + idx)})",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 24),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=12,
        weight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

for ax in axes_flat3[:n_rivers]:
    y_min, y_max = ax.get_ylim()
    y_abs_max = max(abs(y_min), abs(y_max))
    ax.set_ylim(-y_abs_max, y_abs_max)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(y):g}" for y in yticks])
for ax in axes_flat3[n_rivers:]:
    ax.set_visible(False)

plt.savefig(basin_analysis_dir / "all_rivers_monthly_bars_rain_snow.png", dpi=350)
plt.close(fig3)
print(f"  saved to {basin_analysis_dir / 'all_rivers_monthly_bars_rain_snow.png'}")

print("Plotting individual per-river monthly discharge bars")
# Individual per-river monthly bar plots
for river, river_data in monthly_snow_rain_data.items():
    plot_monthly_bars_rain_snow(
        river_data["scenario_snowmelt"],
        river_data["scenario_rain"],
        river_data["control_snowmelt"],
        river_data["control_rain"],
        river,
        save_dir=basin_analysis_dir,
    )
print(f"  saved {n_rivers} individual plots to {basin_analysis_dir}")

print("Plotting seasonal cycle phases (all rivers)")
# One row per river, 3 fixed columns (Snowmelt/Rain/Total). figsize and the
# spacing/label positions set further below are tuned for ~6 rivers and may
# need adjusting if MAIN_RIVERS grows or shrinks substantially.
fig, axes = plt.subplots(n_rivers, 3, figsize=(11, 13), sharex=True)
for i, (river, river_data) in enumerate(monthly_snow_rain_data.items()):
    plot_seasonal_cycle_phases(
        river_data["scenario_snowmelt"],
        river_data["scenario_rain"],
        river_data["control_snowmelt"],
        river_data["control_rain"],
        river,
        axes=axes[i, :],
        show_component_titles=(i == 0),
        show_component_ylabels=False,
        show_river_title=False,
        show_xticklabels=(i == n_rivers - 1),
        show_legend=False,
    )

for idx, ax in enumerate(axes.flatten()):
    panel_label = ax.text(
        0.03,
        0.98,
        f"{chr(97 + idx)})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )
    panel_label.set_bbox(dict(boxstyle="round", facecolor="white", alpha=0.7))

fig.subplots_adjust(
    left=0.09, right=0.98, top=0.93, bottom=0.05, hspace=0.15, wspace=0.2
)

# Eine gemeinsame y-Achsenbeschriftung statt 18 identischer Einzelbeschriftungen
fig.text(
    0.015,
    0.5,
    "Discharge (10³ m³ s⁻¹)",
    va="center",
    ha="center",
    rotation="vertical",
    fontsize=11,
)

# Beckenname als gedrehte Zeilenbeschriftung links außen statt als Titel,
# der mit dem Panel-Label kollidieren wuerde
for i, river in enumerate(monthly_snow_rain_data.keys()):
    pos = axes[i, 0].get_position()
    y_center = (pos.y0 + pos.y1) / 2
    fig.text(
        0.045,
        y_center,
        river,
        va="center",
        ha="center",
        rotation=90,
        fontsize=8,
        fontweight="bold",
    )

# Eine horizontale Legende fuer die ganze Abbildung statt im einzelnen Panel
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=len(labels),
    fontsize=8,
    frameon=False,
)

plt.savefig(basin_analysis_dir / "seasonal_cycle_phases.png", dpi=600)
plt.close(fig)
print(f"  saved to {basin_analysis_dir / 'seasonal_cycle_phases.png'}")

print("\n=== DONE ===")
