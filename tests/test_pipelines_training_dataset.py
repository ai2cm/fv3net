import numpy as np
import pytest
import xarray as xr

from fv3net.pipelines.create_training_data.pipeline import (
    _add_apparent_sources,
    _add_physics_tendencies,
)
from fv3net.pipelines.create_training_data.helpers import (
    _convert_forecast_time_to_timedelta,
)


@pytest.fixture
def test_training_raw_ds():
    centered_coords = {
        "initial_time": [
            np.datetime64("2020-01-01T00:00").astype("M8[ns]"),
            np.datetime64("2020-01-01T00:15").astype("M8[ns]"),
        ],
        "forecast_time": np.array([0.0, 60.0, 120.0]),
        "step": ["after_dynamics", "after_physics"],
    }

    # hi res changes by 0 K, C48 changes by 1 K and 26 K
    T_da = xr.DataArray(
        [
            [[273.0, 274.0, 300.0], [273.0, 275.0, 310.0]],
            [[-273.0, -274.0, -300.0], [-273.0, -275.0, -310.0]],
        ],
        dims=["step", "initial_time", "forecast_time"],
        coords=centered_coords,
    )

    return xr.Dataset({"air_temperature": T_da})


def test__add_apparent_sources(test_training_raw_ds):
    ds = test_training_raw_ds.sel(step="after_dynamics")
    ds = _convert_forecast_time_to_timedelta(ds, "forecast_time")

    train_ds = _add_apparent_sources(
        ds,
        tendency_tstep_onestep=1,
        tendency_tstep_highres=1,
        init_time_dim="initial_time",
        forecast_time_dim="forecast_time",
        var_source_name_map={"air_temperature": "dQ1"},
    )
    assert train_ds.dQ1.values == pytest.approx(1.0 / (15.0 * 60) - 26.0 / 60)


def test__add_physics_tendencies(test_training_raw_ds):
    train_ds = _add_physics_tendencies(
        test_training_raw_ds,
        physics_tendency_names={"air_temperature": "pQ1"},
        forecast_time_dim="forecast_time",
        step_dim="step",
        coord_before_physics="after_dynamics",
        coord_after_physics="after_physics",
    )
    assert train_ds["pQ1"].isel(initial_time=0).values == pytest.approx(
        np.array([-273 * 2 / 60.0, -274 * 2 / 60.0, -300 * 2 / 60])
    )
