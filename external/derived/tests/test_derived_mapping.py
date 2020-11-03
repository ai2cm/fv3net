from typing import Mapping
import numpy as np
import cftime
import xarray as xr
from derived import DerivedMapping

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


def test_DerivedMapping():
    derived_state = DerivedMapping(ds)
    assert isinstance(derived_state["T"], xr.DataArray)


def test_DerivedMapping_cos_zenith():
    derived_state = DerivedMapping(ds)
    output = derived_state["cos_zenith_angle"]
    assert isinstance(output, xr.DataArray)


def test_DerivedMapping_map():
    derived_state = DerivedMapping(ds)
    keys = ["T", "q"]
    map = derived_state.map(keys)
    assert isinstance(derived_state.map(keys), Mapping)
    assert set(keys) == set(map.keys())


def test_DerivedMapping_dataset():
    derived_state = DerivedMapping(ds)
    keys = ["T", "q", "cos_zenith_angle"]
    ds_derived_state = derived_state.dataset(keys)
    assert isinstance(ds_derived_state, xr.Dataset)
    for existing_var in ["T", "q"]:
        np.testing.assert_array_almost_equal(
            ds_derived_state[existing_var], ds[existing_var]
        )
