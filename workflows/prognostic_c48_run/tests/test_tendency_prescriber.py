from datetime import timedelta
import cftime
import numpy as np
import xarray as xr

import fv3gfs.util
from fv3gfs.util.testing import DummyComm

from runtime.transformers.tendency_prescriber import (
    TendencyPrescriber,
    TendencyPrescriberConfig,
)


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
    return fv3gfs.util.CubedSphereCommunicator(
        DummyComm(0, 6, {}),
        fv3gfs.util.CubedSpherePartitioner(fv3gfs.util.TilePartitioner((1, 1))),
    )


def test_tendency_prescriber(state, tmpdir):
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
    override = TendencyPrescriber(
        TendencyPrescriberConfig(path, {"air_temperature": "Q1"}),
        derived_state,
        communicator,
        timestep=2,
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
    np.testing.assert_allclose(
        diags["tendency_of_air_temperature_due_to_tendency_prescriber"].values,
        np.ones((6, 63, 4, 4)),
    )
    assert "some_diag" in diags
