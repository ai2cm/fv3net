import numpy as np
import pytest
import xarray as xr

from fv3net.pipelines.create_training_data.pipeline import _create_train_cols


@pytest.fixture
def test_training_raw_ds():
    centered_coords = {
        "grid_yt": [1],
        "grid_xt": [1],
        "initialization_time": [
            np.datetime64("2020-01-01T00:00").astype("M8[ns]"),
            np.datetime64("2020-01-01T00:15").astype("M8[ns]"),
        ],
        "forecast_time": [
            np.datetime64("2020-01-01T00:00").astype("M8[ns]"),
            np.datetime64("2020-01-01T00:15").astype("M8[ns]"),
        ],
    }
    x_component_edge_coords = {
        "grid_y": [0, 2],
        "grid_xt": [1],
        "initialization_time": [
            np.datetime64("2020-01-01T00:00").astype("M8[ns]"),
            np.datetime64("2020-01-01T00:15").astype("M8[ns]"),
        ],
        "forecast_time": [
            np.datetime64("2020-01-01T00:00").astype("M8[ns]"),
            np.datetime64("2020-01-01T00:15").astype("M8[ns]"),
        ],
    }

    # hi res changes by 0 K, C48 changes by 1 K
    T_da = xr.DataArray(
        [[[[273.0, 274.0], [273.0, 274.0]]]],
        dims=["grid_yt", "grid_xt", "initialization_time", "forecast_time"],
        coords=centered_coords,
    )

    u_da = xr.DataArray(
        [[[[20, 30], [20.0, 30.0]]], [[[30, 40], [30.0, 40.0]]]],
        dims=["grid_y", "grid_xt", "initialization_time", "forecast_time"],
        coords=x_component_edge_coords,
    )

    ds = xr.Dataset({"T": T_da, "sphum": T_da, "u": u_da, "v": u_da})
    return ds


def test__create_train_cols(test_training_raw_ds):
    train_ds = _create_train_cols(test_training_raw_ds, cols_to_keep=["Q1", "QU"])
    assert pytest.approx(train_ds.Q1.values) == -1.0 / (15.0 * 60)
    assert pytest.approx(train_ds.QU.values) == -10.0 / (15.0 * 60)
