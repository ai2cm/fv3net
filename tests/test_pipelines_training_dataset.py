import numpy as np
import pytest
import xarray as xr

from fv3net.pipelines.create_training_data.pipeline import (
    _add_apparent_sources, _preprocess_one_step_data)


@pytest.fixture
def test_training_raw_ds():
    centered_coords = {
        "step": ["begin", "after_physics"],
        "initial_time": [
            np.datetime64("2020-01-01T00:00").astype("M8[ns]"),
            np.datetime64("2020-01-01T00:15").astype("M8[ns]"),
        ],
        "forecast_time": np.array([0.0, 60.0, 120.0]).astype(np.dtype("<m8[s]")),
    }

    # hi res changes by 0 K, C48 changes by 1 K and 6 K
    T_da = xr.DataArray(
        [
            [[273.0, 274.0, 300.0], [273.0, 275.0, 310.0]],
            [[0.0, 1.0, 3.0], [2.0, 3.0, 4.0]],
        ],
        dims=["step", "initial_time", "forecast_time"],
        coords=centered_coords,
    )

    return xr.Dataset({"air_temperature": T_da})


def test__create_train_cols(test_training_raw_ds):
    ds = _preprocess_one_step_data(
        test_training_raw_ds,
        radiation_vars=[],
        forecast_time_dim="forecast_time",
        suffix_coarse_train="train",
        step_time_dim="step",
        coord_begin_step="begin",
    )
    train_ds = _add_apparent_sources(
        ds,
        tendency_tstep_onestep=1,
        tendency_tstep_highres=1,
        init_time_dim="initial_time",
        forecast_time_dim="forecast_time",

        var_source_name_map={"air_temperature": "dQ1"},
    )
    assert train_ds.dQ1.values == pytest.approx(1.0 / (15.0 * 60) - 26.0 / 60)
