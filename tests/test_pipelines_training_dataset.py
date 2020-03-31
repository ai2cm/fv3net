import numpy as np
import pytest
import xarray as xr

from fv3net.pipelines.create_training_data.pipeline import _create_train_cols


@pytest.fixture
def test_training_raw_ds():
    centered_coords = {
        "y": [1],
        "x": [1],
        "initial_time": [
            np.datetime64("2020-01-01T00:00").astype("M8[ns]"),
            np.datetime64("2020-01-01T00:15").astype("M8[ns]"),
        ],
        "forecast_time": [
            np.datetime64("2020-01-01T00:00").astype("M8[ns]"),
            np.datetime64("2020-01-01T00:15").astype("M8[ns]"),
        ],
    }
    x_component_edge_coords = {
        "y_interface": [0, 2],
        "x": [1],
        "initial_time": [
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
        dims=["y", "x", "initial_time", "forecast_time"],
        coords=centered_coords,
    )

    u_da = xr.DataArray(
        [[[[20, 30], [20.0, 30.0]]], [[[30, 40], [30.0, 40.0]]]],
        dims=["y_interface", "x", "initial_time", "forecast_time"],
        coords=x_component_edge_coords,
    )

    ds = xr.Dataset(
        {
            "air_temperature": T_da,
            "specific_humidity": T_da,
            "x_wind": u_da,
            "y_wind": u_da,
        }
    )
    return ds


def test__create_train_cols(test_training_raw_ds):
    train_ds = _create_train_cols(
        test_training_raw_ds, cols_to_keep=["air_temperature", "dQ1", "dQU"]
    )
    assert pytest.approx(train_ds.dQ1.values) == -1.0 / (15.0 * 60)
    assert pytest.approx(train_ds.dQU.values) == -10.0 / (15.0 * 60)
