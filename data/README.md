Input data for `main.py`. Everything except the CESM model output is committed.

## `cesm_output_from_harrison/temp_precip/` (not committed, 262 MB)

CESM monthly output (`TS`, `TSMN`, `TSMX`, `PRECC`, `PRECL`) for the control
run and the 16, 47 and 150 Tg soot-injection scenarios, openly available via
the Open Science Framework (https://doi.org/10.17605/OSF.IO/E28GQ). `main.py`
downloads and extracts it automatically if missing:

| Run | Scenario | Download |
|---|---|---|
| `nw_cntrl_03` | Control | https://osf.io/download/w6qb5/ |
| `nw_targets_04` | 16 Tg | https://osf.io/download/n2vtj/ |
| `nw_targets_05` | 47 Tg | https://osf.io/download/4uysj/ |
| `nw_ur_150_07` | 150 Tg | https://osf.io/download/wnvxf/ |

Archive names: `<run>.TS_TSMN_TSMX_PRECC_PRECL_v2024-12-12.nc.tar.gz`

Harrison, C., Faulkner, W., Coupe, J., Asante, E. K., Bardeen, C., Garza, V.,
Heneghan, R., Jägermeyr, J., Lovenduski, N. S., Robock, A., Rojas, K.,
Scherrer, K., Toon, O. B., Xia, L. (2026): Accessible Climate and Impact Model
Output for Studying the Human and Environmental Impacts of Nuclear Conflict.
Geoscience Data Journal, 13(3): e70076. https://doi.org/10.1002/gdj3.70076

## `natural_earth_land_polygons_110m/`

Natural Earth 1:110m physical land polygons, version 4.1.0 (public domain),
https://www.naturalearthdata.com/downloads/110m-physical-vectors/110m-land/.
Used to build the land mask; `main.py` reads the same file from
https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip.

## `hydrobasins_level4/`

HydroBASINS v1c, standard, level 4, regions `eu`, `as`, `si`, `na`
(https://www.hydrosheds.org/products/hydrobasins), downloaded from
`https://data.hydrosheds.org/file/hydrobasins/standard/hybas_<region>_lev01-12_v1c.zip`
and converted from shapefile to GeoPackage by `load_basin_region()` in
`src/preprocessing/download_data.py` (downloaded 2026-06-30).

Lehner, B., Grill, G. (2013): Global river hydrography and network routing:
baseline data and new approaches to study the world's large river systems.
Hydrological Processes, 27(15): 2171–2186. https://doi.org/10.1002/hyp.9740

## `land_fraction_mask/`

`land_mask_neu.nc`: land fraction (0–1) per grid cell on the CESM grid
(96 lat × 144 lon), computed from the Natural Earth polygons by `create_mask()`
in `src/utilities.py`.
