import cftime
import numpy as np
import pytest
import xarray as xr
from vcm.calc.thermo import (
    GRAVITY,
    pressure_at_interface,
    height_at_interface,
    _interface_to_midpoint,
    dz_and_top_to_phis,
    _add_coords_to_interface_variable,
)
from vcm.calc.calc import local_time
from vcm.cubedsphere.constants import COORD_Z_CENTER, COORD_Z_OUTER


@pytest.mark.parametrize("toa_pressure", [0, 5])
def test_pressure_on_interface(toa_pressure):
    delp = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_CENTER])
    pressure = pressure_at_interface(delp, toa_pressure=toa_pressure)
    xr.testing.assert_allclose(
        pressure.diff(COORD_Z_OUTER), delp.rename({COORD_Z_CENTER: COORD_Z_OUTER})
    )
    np.testing.assert_allclose(pressure.isel({COORD_Z_OUTER: 0}).values, toa_pressure)


@pytest.mark.parametrize("phis_value", [0, 5])
def test_height_on_interface(phis_value):
    dz = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_CENTER])
    phis = xr.DataArray(phis_value)
    height = height_at_interface(dz, phis)
    xr.testing.assert_allclose(
        height.diff(COORD_Z_OUTER), dz.rename({COORD_Z_CENTER: COORD_Z_OUTER})
    )
    np.testing.assert_allclose(
        height.isel({COORD_Z_OUTER: -1}).values, phis_value / GRAVITY
    )


def test__interface_to_midpoint():
    interface = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_OUTER])
    midpoint = _interface_to_midpoint(interface)
    expected = xr.DataArray(np.arange(1.5, 9), dims=[COORD_Z_CENTER])
    xr.testing.assert_allclose(midpoint, expected)


@pytest.mark.parametrize(
    "da_coords, da_out_expected_coords",
    [
        ({"x": range(5)}, {"x": range(5)}),
        ({"z": range(10), "x": range(5)}, {"x": range(5)}),
    ],
)
def test__add_coords_to_interface_variable(da_coords, da_out_expected_coords):
    da = xr.DataArray(np.ones((10, 5)), dims=["z", "x"], coords=da_coords)
    dv = xr.Variable(["z", "x"], np.ones((11, 5)))
    da_out = _add_coords_to_interface_variable(dv, da, dim_center="z")
    da_out_expected = xr.DataArray(
        np.ones((11, 5)), dims=["z", "x"], coords=da_out_expected_coords
    )
    xr.testing.assert_allclose(da_out, da_out_expected)


def test_dz_and_top_to_phis():
    top = 10.0
    dz = np.arange(-4, -1)
    dza = xr.DataArray(dz, dims=[COORD_Z_CENTER])
    phis = dz_and_top_to_phis(top, dza)
    np.testing.assert_allclose(phis.values / GRAVITY, top + np.sum(dz))


def test_solar_time():
    t = xr.DataArray(
        [
            cftime.DatetimeJulian(2020, 1, 1, 0, 0),
            cftime.DatetimeJulian(2020, 1, 1, 0, 0),
            cftime.DatetimeJulian(2020, 1, 1, 0, 0),
            cftime.DatetimeJulian(2020, 1, 1, 0, 0),
            cftime.DatetimeJulian(2020, 1, 1, 6, 0),
            cftime.DatetimeJulian(2020, 1, 1, 6, 0),
        ],
        dims=["x"],
        coords={"x": range(6)},
    )
    lon = xr.DataArray([0, 180, 270, 360, 0, 270], dims=["x"], coords={"x": range(6)})
    ds_solar_test = xr.Dataset({"initialization_time": t, "lon": lon})
    assert np.allclose(local_time(ds_solar_test), [0, 12, 18, 0, 6, 0])

