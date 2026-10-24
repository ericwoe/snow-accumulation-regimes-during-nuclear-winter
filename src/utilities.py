import cartopy.crs as ccrs
import cftime
import numpy as np
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from matplotlib.transforms import offset_copy
from shapely.geometry import box


def create_mask(da, shape):
    """
    Compute the land-covered fraction for each grid cell.

    Arguments:
        da:    xr.DataArray with lat and lon coordinates defining the grid
        shape: geopandas.GeoDataFrame with land polygons (lon -180 to 180)

    Returns:
        xr.DataArray, shape (lat, lon) - land fraction between 0 and 1
    """
    # Combine all land polygons into one geometry (union)
    shape_union = shape.union_all()

    fractions = np.zeros((len(da.lat), len(da.lon)))

    # Half cell size in degrees to define cell boundaries around each centre point
    dlat = float(abs(da.lat[1] - da.lat[0])) / 2
    dlon = float(abs(da.lon[1] - da.lon[0])) / 2

    for i, lat in enumerate(da.lat.values):
        for j, lon in enumerate(da.lon.values):
            # Convert 0–360 longitude to -180–180 to match the shapefile CRS
            lon_centered = lon if lon <= 180 else lon - 360

            # Clip latitude boundaries to ±90° to avoid invalid pole geometries
            lat_lower = max(lat - dlat, -90)
            lat_upper = min(lat + dlat, 90)

            lon_left = lon_centered - dlon
            lon_right = lon_centered + dlon

            # Cell crosses the International Date Line to the right (>180°):
            # split and mirror the right half to the west side of the map
            if lon_right > 180:
                cell_left = box(lon_left, lat_lower, 180, lat_upper)
                cell_right = box(-180, lat_lower, lon_right - 360, lat_upper)
                intersection = (
                    shape_union.intersection(cell_left).area
                    + shape_union.intersection(cell_right).area
                )
                cell_area = cell_left.area + cell_right.area

            # Cell crosses the International Date Line to the left (<-180°):
            # mirror the left half to the east side of the map
            elif lon_left < -180:
                cell_left = box(lon_left + 360, lat_lower, 180, lat_upper)
                cell_right = box(-180, lat_lower, lon_right, lat_upper)
                intersection = (
                    shape_union.intersection(cell_left).area
                    + shape_union.intersection(cell_right).area
                )
                cell_area = cell_left.area + cell_right.area

            else:
                cell = box(lon_left, lat_lower, lon_right, lat_upper)
                intersection = shape_union.intersection(cell).area
                cell_area = cell.area

            fractions[i, j] = intersection / cell_area

    return xr.DataArray(
        fractions,
        coords={"lat": da.lat, "lon": da.lon},
        dims=["lat", "lon"],
        attrs={"long_name": "Land fraction", "units": "None"},
    )


def compute_grid_cell_area(da):
    """
    Compute the area of each grid cell in a regular lat/lon grid.

    Uses the spherical Earth approximation. Cell area varies only with latitude,
    so the result is computed as a 1D array along latitude and then broadcast to
    match the full 2D spatial shape of the input.

    Arguments:
        da: xr.DataArray, shape (lat, lon) - pass a single time slice,
            e.g. ds.snow_storage.isel(time=0)

    Returns:
        xr.DataArray, shape (lat, lon) - grid cell area in square metres
    """
    R = 6371000.0  # Earth radius [m]
    dlat = abs(da.lat[1] - da.lat[0])
    dlon = abs(da.lon[1] - da.lon[0])
    lat_rad = np.deg2rad(da.lat)
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)

    lat_upper = np.clip(lat_rad + dlat_rad / 2, -np.pi / 2, np.pi / 2)
    lat_lower = np.clip(lat_rad - dlat_rad / 2, -np.pi / 2, np.pi / 2)

    # Area of a latitude band of width dlon, between lat_lower and lat_upper
    area_1d = R**2 * dlon_rad * np.abs(np.sin(lat_upper) - np.sin(lat_lower))

    # Expand from 1D (latitude only) to 2D (latitude x longitude)
    cell_area = area_1d.broadcast_like(da)
    cell_area.name = "cell_area"
    cell_area.attrs["units"] = "m2"
    return cell_area


def change_time(ds):
    """
    Shift the time coordinate of a dataset back by one month, in place.

    The input datasets are assumed to use a no-leap calendar where each
    timestamp marks the end of the averaging period (e.g., a value labeled
    February actually represents January). This function relabels each
    timestamp to the preceding month, so that month 1 corresponds to the
    first full month after the start of the simulation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a "time" coordinate using a no-leap cftime calendar.
        The "time" coordinate is modified in place.
    """
    new_time = [
        cftime.DatetimeNoLeap(
            t.year if t.month > 1 else t.year - 1,
            t.month - 1 if t.month > 1 else 12,
            1,
        )
        for t in ds.time.values
    ]
    ds.coords["time"] = new_time


def add_lat_lon_gridlines(ax, linewidth, fontsize, latitudes=(-60, -30, 0, 30, 60)):
    """
    Draw latitude/longitude gridlines labelled on the left and bottom edge only,
    as required by the AGU figure guidelines.

    Cartopy labels gridlines that meet the curved side of a Robinson map as "geo"
    labels on both sides, so the latitude labels are placed manually at the left edge.

    Arguments:
        ax:        cartopy GeoAxes in Robinson projection
        linewidth: float - gridline width [pt]
        fontsize:  float - label font size [pt]
        latitudes: tuple of float - latitudes of the gridlines and labels [°]
    """
    gl = ax.gridlines(
        draw_labels={"bottom": "x"},
        ylocs=latitudes,
        linewidth=linewidth,
        color="grey",
        alpha=0.5,
        linestyle="--",
    )
    # Draw above ocean (10) and coastlines/borders (11) so lines also show over the ocean.
    # The Gridliner is its own artist, so its zorder (not the lines') sets the draw order.
    gl.set_zorder(12)
    gl.xlabel_style = {"size": fontsize}

    # Anchor at lon=-180 (left map edge) and shift a few points further left
    left_edge = offset_copy(
        ccrs.PlateCarree()._as_mpl_transform(ax), fig=ax.figure, x=-4, units="points"
    )
    for lat in latitudes:
        ax.text(
            -180,
            lat,
            LATITUDE_FORMATTER(lat),
            transform=left_edge,
            ha="right",
            va="center",
            size=fontsize,
        )
