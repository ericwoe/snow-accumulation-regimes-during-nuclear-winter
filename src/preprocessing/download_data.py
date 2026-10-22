"""
Download and cache raw input data used by the pipeline: the CESM model
output archives (from OSF) and the HydroBasins shapefiles (from
HydroSHEDS) used for the river-basin discharge analysis.
"""

import tarfile
import urllib.request
from pathlib import Path

import geopandas as gpd


def download_model_output(name, config):
    """Download and extract a scenario's model output from OSF if not already present."""
    data_dir = Path(config["data_dir"])
    if data_dir.exists() and any(data_dir.glob(config["pattern"])):
        print(f"  {name}: data already present, skipping download")
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / config["osf_archive"]
    print(f"  {name}: downloading from {config['osf_url']}, storing in {data_dir}")
    urllib.request.urlretrieve(config["osf_url"], archive_path)
    print(f"  {name}: extracting...")
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            # The archive preserves the original server path; strip all directory
            # components so files land directly in data_dir.
            filename = Path(member.name).name
            if filename.endswith(".nc"):
                member.name = filename
                tar.extract(member, data_dir)
    archive_path.unlink()
    print(f"  {name}: done")


def load_basin_region(region, cache_dir, urls):
    """Load a HydroBasins level-4 region, downloading and caching it if needed."""
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"hybas_{region}_lev04.gpkg"
    if cache_path.exists():
        print(f"Loading HydroBasins region '{region}' from cache")
        return gpd.read_file(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vsi_path = f"/vsizip//vsicurl/{urls[region]}/hybas_{region}_lev04_v1c.shp"
    print(
        f"Downloading HydroBasins region '{region}' from {urls[region]}, "
        f"caching to {cache_path}"
    )
    gdf = gpd.read_file(vsi_path)
    gdf.to_file(cache_path, driver="GPKG")
    return gdf
