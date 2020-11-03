from typing import Mapping
import numpy as np
import cftime
import xarray as xr
from derived import DerivedState

nt, nx, ny = 3, 2, 1
ds = xr.Dataset(
    {
        "lat": xr.DataArray(np.random.rand(ny, nx), dims=["y", "x"]),
        "lon": xr.DataArray(np.random.rand(ny, nx), dims=["y", "x"]),
        "T": xr.DataArray(
            np.random.rand(ny, nx, nt),
            dims=["y", "x", "time"],
            coords={"time": [cftime.DatetimeJulian(2016, 1, 1) for n in range(nt)]},
        ),
        "q": xr.DataArray(
            np.random.rand(ny, nx, nt),
            dims=["y", "x", "time"],
            coords={"time": [cftime.DatetimeJulian(2016, 1, 1) for n in range(nt)]},
        ),
    }
)


def test_DerivedState():
    derived_state = DerivedState(ds)
    assert isinstance(derived_state["T"], xr.DataArray)


def test_DerivedState_cos_zenith():
    derived_state = DerivedState(ds)
    output = derived_state["cos_zenith_angle"]
    assert isinstance(output, xr.DataArray)


def test_DerivedState_map():
    derived_state = DerivedState(ds)
    keys = ["T", "q"]
    map = derived_state.map(keys)
    assert isinstance(derived_state.map(keys), Mapping)
    assert set(keys) == set(map.keys())


def test_DerivedState_dataset():
    derived_state = DerivedState(ds)
    keys = ["T", "q", "cos_zenith_angle"]
    ds_derived_state = derived_state.dataset(keys)
    assert isinstance(ds_derived_state, xr.Dataset)
    for existing_var in ["T", "q"]:
        np.testing.assert_array_almost_equal(
            ds_derived_state[existing_var], ds[existing_var]
        )
