from datetime import timedelta
import joblib
import cftime
import numpy as np
import xarray as xr
import dacite
import pytest

import pace.util
from pace.util.testing import DummyComm

import loaders
from runtime.transformers.tendency_prescriber import (
    TendencyPrescriber,
    TendencyPrescriberConfig,
)
from runtime.factories import _get_time_lookup_function


class MockDerivedState:
    def __init__(self, state: xr.Dataset, time: cftime.DatetimeJulian):
        self._state = state
        self._time = time

    @property
    def time(self) -> cftime.DatetimeJulian:
        return self._time

    def __getitem__(self, key: str) -> xr.DataArray:
        return self._state[key]

    def __setitem__(self, key: str, value: xr.DataArray):
        self._state[key] = value


def _get_tendencies(time: cftime.DatetimeJulian) -> xr.Dataset:
    tendency_da = xr.DataArray(
        data=np.ones((6, 5, 63, 4, 4)),
        dims=["tile", "time", "z", "y", "x"],
        coords={"time": [time + timedelta(minutes=n) for n in range(0, 5 * 15, 15)]},
        attrs={"units": "K/s"},
    )
    return xr.Dataset({"Q1": tendency_da}).chunk({"time": 2})  # type: ignore


def _get_derived_state(ds, time):
    tiled_state = xr.concat([ds.assign_coords(tile=t) for t in range(6)], dim="tile")
    return MockDerivedState(tiled_state, time)


def _get_dummy_comm():
    return pace.util.CubedSphereCommunicator(
        DummyComm(0, 6, {}),
        pace.util.CubedSpherePartitioner(pace.util.TilePartitioner((1, 1))),
    )


def test_tendency_prescriber(state, tmpdir, regtest):
    time = cftime.DatetimeJulian(2016, 8, 1)
    path = str(tmpdir.join("tendencies.zarr"))
    tendencies = _get_tendencies(time)
    tendencies.to_zarr(path, consolidated=True)
    derived_state = _get_derived_state(state, time)
    derived_state_copy = _get_derived_state(state, time)
    communicator = _get_dummy_comm()
    diagnostic_variables = [
        "tendency_of_air_temperature_due_to_override",
        "specific_humidity",
    ]
    config = {
        "mapper_config": {"function": "open_zarr", "kwargs": {"data_path": path}},
        "variables": {"air_temperature": "Q1"},
    }
    prescriber_config = dacite.from_dict(TendencyPrescriberConfig, config)
    timestep = 2
    mapper = prescriber_config.mapper_config.load_mapper()
    mapper_func = _get_time_lookup_function(
        mapper, list(prescriber_config.variables.values()), initial_time=None,
    )
    override = TendencyPrescriber(
        derived_state,
        communicator,
        timestep,
        prescriber_config.variables,
        mapper_func,
        diagnostic_variables=diagnostic_variables,
    )

    def add_one():
        derived_state["air_temperature"] = derived_state["air_temperature"] + 1
        derived_state["specific_humidity"] = derived_state["specific_humidity"] + 1
        return {"some_diag": derived_state["specific_humidity"]}

    diags = override(add_one)()

    xr.testing.assert_identical(
        derived_state["specific_humidity"], derived_state_copy["specific_humidity"] + 1,
    )
    xr.testing.assert_identical(
        derived_state["air_temperature"],
        (derived_state_copy["air_temperature"] + 2).assign_attrs(units="degK"),
    )
    expected_monitored_tendency = (
        tendencies.isel(time=0).drop("time").Q1.assign_coords(tile=range(6))
    )
    xr.testing.assert_allclose(
        diags["tendency_of_air_temperature_due_to_tendency_prescriber"],
        expected_monitored_tendency,
    )
    for variable in sorted(diags):
        print(variable, joblib.hash(diags[variable].values), file=regtest)


@pytest.mark.parametrize(
    ["initial_time", "time_substep", "error"],
    [
        pytest.param("20160801.000000", 450, False, id="interpolate"),
        pytest.param(None, 900, False, id="no_interpolate"),
        pytest.param(None, 450, True, id="substep_without_interpolate_error"),
    ],
)
def test__get_time_lookup_function(tmpdir, initial_time, time_substep, error):
    time = cftime.DatetimeJulian(2016, 8, 1)
    path = str(tmpdir.join("tendencies.zarr"))
    tendencies = _get_tendencies(time)
    tendencies.to_zarr(path, consolidated=True)
    mapper_config = loaders.MapperConfig(
        function="open_zarr", kwargs={"data_path": path}
    )
    mapper = mapper_config.load_mapper()
    time_lookup_function = _get_time_lookup_function(mapper, ["Q1"], initial_time, 900)
    if error:
        with pytest.raises(KeyError):
            time_lookup_function(time + timedelta(seconds=time_substep))
    else:
        state = time_lookup_function(time + timedelta(seconds=time_substep))
        assert isinstance(state["Q1"], xr.DataArray)
