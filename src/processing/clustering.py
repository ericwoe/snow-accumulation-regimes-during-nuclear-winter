import xarray as xr
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from datetime import datetime
import os
import pickle
import pandas as pd


def prepare_time_series(da: xr.DataArray) -> np.ndarray:
    """
    Prepares the time series data for tslearn clustering.

    Arguments:
        da: xarray.DataArray, shape (time, lat, lon)

    Returns:
        timeseries_object: np.ndarray,
                           shape (n_land_cells, time_steps, 1)
    """

    # Reorder Dimensions to (lat, lon, time)
    da_t = da.transpose("lat", "lon", "time")

    # Create land mask (assuming NaNs represent ocean)
    land_mask = ~np.isnan(da.isel(time=0))

    # Extract time series for land cells only
    timeseries_2d = da_t.values[land_mask.values]
    # shape: (n_land_cells, time_steps)

    # Add feature dimension for tslearn
    timeseries_3d = timeseries_2d[..., np.newaxis]
    # shape: (n_land_cells, time_steps, 1)

    return timeseries_3d


def time_series_analysis(timeseries_object: np.ndarray, n_clusters: int):
    """
    Does time series analysis on the timeseries objects.
    All the time serieses are clustered based on their
    overall shape using k-means
    Inspired by this article:
    https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook

    Arguments:
        da: xarray.DataArray, shape (time, lat, lon)
        n_clusters: int - the number of clusters to use

    Returns:
        labels: np.ndarray, shape (n_land_cells) - the labels for each time series
        km: TimeSeriesKMeans - the k-means object
    """
    # Normalize the data
    print("Apply MinMax-Scaling method")
    scaler = TimeSeriesScalerMinMax()
    timeseries_scaled = scaler.fit_transform(timeseries_object)

    # Perform time series k-means clustering
    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        n_jobs=-1,  # use all cpu cores
        max_iter=50,
        random_state=42,
        verbose=1,
    )
    labels = km.fit_predict(timeseries_scaled)
    return labels, km


def elbow_method(timeseries, max_clusters, save_dir):
    """
    Finds the optimal number of clusters using the elbow method
    https://predictivehacks.com/k-means-elbow-method-code-for-python/
    Arguments:
        timeseries: np.ndarray, shape (n_land_cells, time_steps, 1)
        max_clusters: int - the maximum number of clusters to try (inclusive)
        save_dir: path-like - directory where results (CSVs, labels, models) are saved
    Returns:
        inertias_df: DataFrame with inertias for each k
    """
    checkpoint_dir = save_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_inertias = os.path.join(checkpoint_dir, "inertias.csv")

    # Load existing results if available
    if os.path.exists(checkpoint_inertias):
        print(f"Loading existing Inertias from {checkpoint_inertias}")
        inertias_df = pd.read_csv(checkpoint_inertias, sep=";", index_col=0)
        inertias = {
            int(k): v for k, v in inertias_df.to_dict()[inertias_df.columns[0]].items()
        }
    else:
        inertias = {}

    # Normalize once
    print("Apply MinMax-Scaling method")
    scaler = TimeSeriesScalerMinMax()
    timeseries_scaled = scaler.fit_transform(timeseries)

    # Try different k values
    for i in range(2, max_clusters + 1):
        if i in inertias:
            print(f"k={i} already computed, skipping...")
            continue

        print(f"\n{'='*60}")
        begin = datetime.now()
        print(f"{begin.strftime('%H:%M:%S')} - Trying {i} clusters")
        print(f"{'='*60}")

        # Train model
        km = TimeSeriesKMeans(
            n_clusters=i,
            metric="dtw",
            n_jobs=-1,
            max_iter=50,
            random_state=42,
            verbose=1,
        )
        labels = km.fit_predict(timeseries_scaled)

        # Save inertia
        inertias[i] = km.inertia_

        print(f"✓ k={i} finished!")
        print(f"  Inertia: {km.inertia_:.2f}")

        # Save labels and model
        labels_path = os.path.join(checkpoint_dir, f"{i}_cluster_labels.npy")
        model_path = os.path.join(checkpoint_dir, f"kmeans_model_{i}.pkl")
        np.save(labels_path, labels)
        with open(model_path, "wb") as f:
            pickle.dump(km, f)

        # Save inertias CSV after each k
        inertias_df = pd.DataFrame.from_dict(
            inertias, orient="index", columns=["inertia"]
        )
        inertias_df.index.name = "k"
        inertias_df.to_csv(checkpoint_inertias, sep=";")

        print("→ Results saved")
        print(f"finished at {datetime.now().strftime('%H:%M:%S')}")

    print(f"\n{'='*60}")
    print("All Clusterings finished!")
    print(f"{'='*60}")

    return inertias_df


if __name__ == "__main__":
    ds_47 = xr.open_dataset("./results/47/snow_47.nc")
    ds_control = xr.open_dataset("./results/Control/snow_control.nc")

    da = ds_control.snow_storage
    timeseries = prepare_time_series(da)
    print(f"Shape of Timeseries: {timeseries.shape}")
    # Subset
    subset_size = int(0.3 * timeseries.shape[0])
    indices = np.random.choice(timeseries.shape[0], subset_size, replace=False)
    timeseries_subset = timeseries[indices]
    print(f"Shape of Subset: {timeseries_subset.shape}")

    elbow_method(
        timeseries_subset,
        max_clusters=10,
        save_dir="./results/Control/clustering/elbow_silhouette",
    )
