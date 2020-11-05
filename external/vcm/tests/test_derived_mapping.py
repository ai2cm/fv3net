import cftime
import numpy as np
import pytest
from typing import Mapping
import xarray as xr

from vcm import DerivedMapping


@pytest.fixture(params=[(3, 2, 1)])
def txy_dims(request) -> str:
    return request.param


@pytest.fixture()
def ds(txy_dims):
    nt, nx, ny = txy_dims
    return xr.Dataset(
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


def test_DerivedMapping(ds):
    derived_state = DerivedMapping(ds)
    assert isinstance(derived_state["T"], xr.DataArray)


def test_DerivedMapping_cos_zenith(ds):
    derived_state = DerivedMapping(ds)
    output = derived_state["cos_zenith_angle"]
    assert isinstance(output, xr.DataArray)


def test_DerivedMapping__data_arrays(ds):
    derived_state = DerivedMapping(ds)
    keys = ["T", "q"]
    data_arrays = derived_state._data_arrays(keys)
    assert isinstance(data_arrays, Mapping)
    assert set(keys) == set(data_arrays.keys())


def test_DerivedMapping_dataset(ds):
    derived_state = DerivedMapping(ds)
    keys = ["T", "q", "cos_zenith_angle"]
    ds_derived_state = derived_state.dataset(keys)
    assert isinstance(ds_derived_state, xr.Dataset)
    for existing_var in ["T", "q"]:
        np.testing.assert_array_almost_equal(
            ds_derived_state[existing_var], ds[existing_var]
        )


def test_DerivedMapping_unregistered(ds):
    derived_state = DerivedMapping(ds)
    with pytest.raises(KeyError):
        derived_state["latent_heat_flux"]


@pytest.mark.parametrize(
    "txy_dims, drop_time",
    [
        pytest.param((3, 2, 1), False, id="multiple_times"),
        pytest.param((1, 2, 1), True, id="single_time"),
    ],
    indirect=["txy_dims"],
)
def test_cos_z_result_time_dims(ds, txy_dims, drop_time):
    derived_state = DerivedMapping(ds)
    output = derived_state["cos_zenith_angle"]
    if drop_time:
        assert "time" not in output.dims
    else:
        assert "time" in output.dims
        assert len(output["time"]) == txy_dims[0]
