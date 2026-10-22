import numpy as np
import pytest
from snow_model import calculate_snow_dynamics


def test_output_shape():
    precip = np.zeros((12, 3, 3), dtype=np.float32)
    t_mean = np.zeros((12, 3, 3), dtype=np.float32)
    days = np.ones(12, dtype=np.float32) * 30

    storage, melt = calculate_snow_dynamics(precip, t_mean, days)

    assert storage.shape == (12, 3, 3)
    assert melt.shape == (12, 3, 3)


def test_melt_never_exceeds_storage():
    precip = np.array([[[50.0]], [[0.0]]], dtype=np.float32)
    t_mean = np.array([[[-5.0]], [[20.0]]], dtype=np.float32)
    days = np.array([31, 31], dtype=np.float32)

    storage, melt = calculate_snow_dynamics(precip, t_mean, days)

    assert melt[1, 0, 0] <= storage[0, 0, 0]
    assert storage[1, 0, 0] >= 0.0


def test_degree_day_formula():
    precip = np.array([[[100.0]], [[0.0]]], dtype=np.float32)
    t_mean = np.array([[[-5.0]], [[5.0]]], dtype=np.float32)
    days = np.array([31, 10], dtype=np.float32)

    storage, melt = calculate_snow_dynamics(precip, t_mean, days)

    potential_melt = 10 * 4.0 * (5.0 - 0.0)  # = 200mm
    available_snow = storage[0, 0, 0]  # = 100mm
    expected_melt = min(potential_melt, available_snow)  # = 100mm

    np.testing.assert_allclose(melt[1, 0, 0], expected_melt, rtol=1e-4)


def test_spatial_independence():
    # Pixel 0: kalt (Schnee), Pixel 1: warm (kein Schnee)
    precip = np.array([[[50.0, 50.0]]], dtype=np.float32)
    t_mean = np.array([[[-5.0, 5.0]]], dtype=np.float32)
    days = np.array([31], dtype=np.float32)

    storage, melt = calculate_snow_dynamics(precip, t_mean, days)

    assert storage[0, 0, 0] == 50.0  # kalter Pixel: Schnee
    assert storage[0, 0, 1] == 0.0  # warmer Pixel: kein Schnee


def test_storage_mass_balance():
    precip = np.array([[[30.0]], [[20.0]], [[0.0]]], dtype=np.float32)
    t_mean = np.array([[[-5.0]], [[-3.0]], [[2.0]]], dtype=np.float32)
    days = np.array([31, 28, 31], dtype=np.float32)

    storage, melt = calculate_snow_dynamics(precip, t_mean, days)

    t_thresh = 1.7  # aus dem Modell

    for t in range(1, 3):
        # Neuer Schnee nur wenn T <= t_thresh
        new_snow = precip[t, 0, 0] if t_mean[t, 0, 0] <= t_thresh else 0.0
        expected_storage = storage[t - 1, 0, 0] + new_snow - melt[t, 0, 0]

        assert storage[t, 0, 0] >= 0.0
        assert storage[t, 0, 0] == pytest.approx(expected_storage, abs=1e-3)
