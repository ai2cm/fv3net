from datetime import timedelta
import cftime
import numpy as np
import xarray as xr

import fv3gfs.util
from fv3gfs.util.testing import DummyComm

from runtime.overrider import DatasetCachedByChunk, OverriderAdapter, OverriderConfig

da = xr.DataArray(
    np.arange(11 * 4 * 6).reshape((11, 4, 6)),
    dims=["time", "x", "y"],
    coords={"time": np.arange(2000, 2011)},
).chunk({"time": 5, "x": 4, "y": 4})
ds = xr.Dataset({"var": da})


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


def _get_tendency_ds(time):
    tendency_da = xr.DataArray(
        data=np.ones((6, 5, 63, 4, 4)),
        dims=["tile", "time", "z", "y", "x"],
        coords={"time": [time + timedelta(minutes=n) for n in range(0, 5 * 15, 15)]},
        attrs={"units": "K/s"},
    )
    return xr.Dataset({"Q1": tendency_da}).chunk({"time": 2})


def test_dataset_caching():
    cached = DatasetCachedByChunk(ds, "time")
    ds_year_2000 = cached.load(2000)
    xr.testing.assert_identical(ds_year_2000, ds.sel(time=2000))
    assert cached._load_chunk.cache_info().hits == 0
    cached.load(2001)
    assert cached._load_chunk.cache_info().hits == 1


def test_overrider(state, tmpdir):
    name = "add_one"
    time = cftime.DatetimeJulian(2016, 8, 1)
    path = str(tmpdir.join("tendencies.zarr"))
    ds = _get_tendency_ds(time)
    ds.to_zarr(path, consolidated=True)
    state, _ = xr.broadcast(
        state[
            [
                "air_temperature",
                "specific_humidity",
                "pressure_thickness_of_atmospheric_layer",
            ]
        ],
        ds.Q1.isel(time=0),
    )
    state = MockDerivedState(state, time)
    state_copy = MockDerivedState(state._state.copy(deep=True), time)
    communicator = fv3gfs.util.CubedSphereCommunicator(
        DummyComm(0, 6, {}),
        fv3gfs.util.CubedSpherePartitioner(fv3gfs.util.TilePartitioner((1, 1))),
    )
    diagnostic_variables = [
        f"tendency_of_air_temperature_due_to_{name}",
        "tendency_of_air_temperature_due_to_override",
        "specific_humidity",
    ]
    override = OverriderAdapter(
        OverriderConfig(path, {"air_temperature": "Q1"}),
        state,
        communicator,
        timestep=2,
        diagnostic_variables=diagnostic_variables,
    )

    def add_one_to_temperature():
        state["air_temperature"] = state["air_temperature"] + 1
        return {"some_diag": state["specific_humidity"]}

    _ = override("add_one", add_one_to_temperature)()

    xr.testing.assert_identical(
        state["specific_humidity"], state_copy["specific_humidity"]
    )
    xr.testing.assert_identical(
        state["air_temperature"],
        (state_copy["air_temperature"] + 2).assign_attrs(units="degK"),
    )
