from datetime import timedelta
import joblib
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
).chunk(
    {"time": 5, "x": 4, "y": 4}  # type: ignore
)
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


def _get_derived_state(ds, time):
    tiled_state = xr.concat([ds.assign_coords(tile=t) for t in range(6)], dim="tile")
    return MockDerivedState(tiled_state, time)


def _get_dummy_comm():
    return fv3gfs.util.CubedSphereCommunicator(
        DummyComm(0, 6, {}),
        fv3gfs.util.CubedSpherePartitioner(fv3gfs.util.TilePartitioner((1, 1))),
    )


def test_dataset_caching():
    cached = DatasetCachedByChunk(ds, "time")
    ds_year_2000 = cached.load(2000)
    xr.testing.assert_identical(ds_year_2000, ds.sel(time=2000))
    assert cached._load_chunk.cache_info().hits == 0
    cached.load(2001)
    assert cached._load_chunk.cache_info().hits == 1


def test_overrider(state, tmpdir, regtest):
    name = "add_one"
    time = cftime.DatetimeJulian(2016, 8, 1)
    path = str(tmpdir.join("tendencies.zarr"))
    tendencies = _get_tendency_ds(time)
    tendencies.to_zarr(path, consolidated=True)
    derived_state = _get_derived_state(state, time)
    derived_state_copy = _get_derived_state(state, time)
    communicator = _get_dummy_comm()
    diagnostic_variables = [
        f"tendency_of_air_temperature_due_to_{name}",
        "tendency_of_air_temperature_due_to_override",
        "specific_humidity",
    ]
    override = OverriderAdapter(
        OverriderConfig(path, {"air_temperature": "Q1"}),
        derived_state,
        communicator,
        timestep=2,
        diagnostic_variables=diagnostic_variables,
    )

    def add_one():
        derived_state["air_temperature"] = derived_state["air_temperature"] + 1
        derived_state["specific_humidity"] = derived_state["specific_humidity"] + 1
        return {"some_diag": derived_state["specific_humidity"]}

    diags = override(name, add_one)()

    xr.testing.assert_identical(
        derived_state["specific_humidity"], derived_state_copy["specific_humidity"] + 1,
    )
    xr.testing.assert_identical(
        derived_state["air_temperature"],
        (derived_state_copy["air_temperature"] + 2).assign_attrs(units="degK"),
    )

    for variable in sorted(diags):
        print(variable, joblib.hash(diags[variable]), file=regtest)
