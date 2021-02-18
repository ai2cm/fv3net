import cftime
import numpy as np
import pytest
from typing import Mapping
import xarray as xr

from vcm import DerivedMapping
from vcm.derived_mapping import _get_datetime_attr

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
        "dQu": xr.DataArray(
            np.ones((ny, nx, nt)),
            dims=["y", "x", "time"],
            coords={"time": [cftime.DatetimeJulian(2016, 1, 1) for n in range(nt)]},
        ),
    }
)


@pytest.mark.parametrize(
    "dQu, dQv, eastward, northward, projection",
    [
        pytest.param(1.0, 0.0, 1.0, 0.0, 1.0, id="parallel, east wind"),
        pytest.param(1.0, 0.0, -1.0, 0.0, -1.0, id="antiparallel, east wind"),
        pytest.param(1.0, 1.0, 1.0, 1.0, 1, id="parallel, 45 deg NE wind"),
        pytest.param(1.0, 0.0, 1.0, 1.0, np.sqrt(2), id="45 deg CCW"),
        pytest.param(-1.0, 0.0, 1.0, 1.0, -np.sqrt(2), id="135 deg CCW"),
    ],
)
def test_horizontal_wind_tendency_parallel_to_horizontal_wind(
    dQu, dQv, eastward, northward, projection
):
    data = xr.Dataset(
        {
            "dQu": xr.DataArray([dQu], dims=["x"]),
            "dQv": xr.DataArray([dQv], dims=["x"]),
            "eastward_wind": xr.DataArray([eastward], dims=["x"]),
            "northward_wind": xr.DataArray([northward], dims=["x"]),
        }
    )
    derived_mapping = DerivedMapping(data)
    assert pytest.approx(
        derived_mapping[
            "horizontal_wind_tendency_parallel_to_horizontal_wind"
        ].values.item(),
        projection,
    )


def test_wind_tendency_derived():
    # dQu/dQv must be calculated from dQxwind, dQywind
    rotation = {
        var: xr.DataArray(np.zeros((ny, nx)), dims=["y", "x"])
        for var in [
            "eastward_wind_u_coeff",
            "eastward_wind_v_coeff",
            "northward_wind_u_coeff",
            "northward_wind_v_coeff",
        ]
    }
    data = xr.Dataset(
        {
            "dQxwind": xr.DataArray(np.ones((ny + 1, nx)), dims=["y_interface", "x"]),
            "dQywind": xr.DataArray(np.ones((ny, nx + 1)), dims=["y", "x_interface"]),
            **rotation,
        }
    )

    derived_mapping = DerivedMapping(data)
    dQu = derived_mapping["dQu"]
    dQv = derived_mapping["dQv"]
    np.testing.assert_array_almost_equal(dQu, 0.0)
    np.testing.assert_array_almost_equal(dQv, 0.0)


def test_wind_tendency_nonderived():
    # dQu/dQv already exist in data
    derived_mapping = DerivedMapping(ds)
    dQu = derived_mapping["dQu"]
    np.testing.assert_array_almost_equal(dQu, 1.0)


def test_DerivedMapping():
    derived_state = DerivedMapping(ds)
    assert isinstance(derived_state["T"], xr.DataArray)


def test_DerivedMapping_cos_zenith():
    derived_state = DerivedMapping(ds)
    output = derived_state["cos_zenith_angle"]
    assert isinstance(output, xr.DataArray)


def test_DerivedMapping__data_arrays():
    derived_state = DerivedMapping(ds)
    keys = ["T", "q"]
    data_arrays = derived_state._data_arrays(keys)
    assert isinstance(data_arrays, Mapping)
    assert set(keys) == set(data_arrays.keys())


def test_DerivedMapping_dataset():
    derived_state = DerivedMapping(ds)
    keys = ["T", "q", "cos_zenith_angle"]
    ds_derived_state = derived_state.dataset(keys)
    assert isinstance(ds_derived_state, xr.Dataset)
    for existing_var in ["T", "q"]:
        np.testing.assert_array_almost_equal(
            ds_derived_state[existing_var], ds[existing_var]
        )


def test_DerivedMapping_unregistered():
    derived_state = DerivedMapping(ds)
    with pytest.raises(KeyError):
        derived_state["latent_heat_flux"]


def test__get_datetime_attr():
    ds = xr.Dataset(
        {
            "time": xr.DataArray(
                data=[
                    cftime.DatetimeJulian(2016, 7, 1),
                    cftime.DatetimeJulian(2016, 12, 31),
                ],
                dims=["time"],
            )
        }
    )

    days = _get_datetime_attr(ds, "dayofyr")
    np.testing.assert_equal(days.values, np.array([183, 366]))
    months = _get_datetime_attr(ds, "month")
    np.testing.assert_equal(months.values, np.array([7, 12]))


def test_cos_sin_derived():
    ds = xr.Dataset(
        {
            "time": xr.DataArray(
                data=[
                    cftime.DatetimeJulian(2016, 7, 1),
                    cftime.DatetimeJulian(2016, 6, 1),
                ],
                dims=["time"],
            ),
            "longitude": xr.DataArray(data=[np.pi]),
        }
    )

    derived_state = DerivedMapping(ds)
    # almost equal because sin op gives order 1e-16 rather than 0
    np.testing.assert_almost_equal(derived_state["cos_day"].values[0], -1)
    np.testing.assert_almost_equal(derived_state["sin_day"].values[0], 0)
    np.testing.assert_almost_equal(derived_state["cos_month"].values[1], -1)
    np.testing.assert_almost_equal(derived_state["sin_month"].values[1], 0)
    np.testing.assert_almost_equal(derived_state["cos_lon"].values[0], -1)
    np.testing.assert_almost_equal(derived_state["sin_lon"].values[0], 0)
