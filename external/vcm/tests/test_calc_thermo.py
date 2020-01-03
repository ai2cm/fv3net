import numpy as np
import pytest
import xarray as xr
from vcm.calc.thermo import (
    GRAVITY,
    pressure_at_interface,
    height_at_interface,
    _interface_to_midpoint,
    dz_and_top_to_phis,
    _add_coords,
)
from vcm.cubedsphere.constants import COORD_Z_CENTER, COORD_Z_OUTER


@pytest.mark.parametrize("toa_pressure", [0, 5])
def test_pressure_on_interface(toa_pressure):
    delp = xr.DataArray(
        np.arange(1, 10),
        dims=[COORD_Z_CENTER],
        coords={COORD_Z_CENTER: np.arange(1, 10)},
    )
    pressure = pressure_at_interface(delp, toa_pressure=toa_pressure)
    pressure_expected = xr.DataArray(
        np.cumsum(np.concatenate(([toa_pressure], delp.values))),
        dims=[COORD_Z_OUTER],
        coords={COORD_Z_OUTER: np.arange(1, 11)},
    )
    xr.testing.assert_allclose(pressure, pressure_expected)


@pytest.mark.parametrize("phis_value", [0, 5])
def test_height_on_interface(phis_value):
    dz = xr.DataArray(
        np.arange(1, 10),
        dims=[COORD_Z_CENTER],
        coords={COORD_Z_CENTER: np.arange(1, 10)},
    )
    phis = xr.DataArray(phis_value)
    height = height_at_interface(dz, phis)
    height_expected = xr.DataArray(
        np.cumsum(np.concatenate(([phis_value / GRAVITY], -dz.values[::-1])))[::-1],
        dims=[COORD_Z_OUTER],
        coords={COORD_Z_OUTER: np.arange(1, 11)},
    )
    xr.testing.assert_allclose(height, height_expected)


@pytest.mark.parametrize("interface_coords", [None, {COORD_Z_OUTER: np.arange(1, 10)}])
def test__interface_to_midpoint(interface_coords):
    interface = xr.DataArray(
        np.arange(1, 10), dims=[COORD_Z_OUTER], coords=interface_coords
    )
    midpoint = _interface_to_midpoint(interface)
    if interface_coords is None:
        midpoint_coords = None
    else:
        midpoint_coords = {COORD_Z_CENTER: interface_coords[COORD_Z_OUTER][:-1]}
    expected = xr.DataArray(
        np.arange(1.5, 9), dims=[COORD_Z_CENTER], coords=midpoint_coords
    )
    xr.testing.assert_allclose(midpoint, expected)


@pytest.mark.parametrize("z_coord", [None, {COORD_Z_CENTER: np.arange(1, 10)}])
def test__add_coords(z_coord):
    da = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_CENTER], coords=z_coord)
    dv = xr.Variable([COORD_Z_CENTER], np.arange(1, 11))
    da_out = _add_coords(dv, da, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER)
    if z_coord is None:
        output_coord = None
    else:
        output_coord = {COORD_Z_OUTER: np.arange(1, 11)}
    da_out_expected = xr.DataArray(
        np.arange(1, 11), dims=[COORD_Z_OUTER], coords=output_coord
    )
    xr.testing.assert_allclose(da_out, da_out_expected)


def test_dz_and_top_to_phis():
    top = 10.0
    dz = np.arange(-4, -1)
    dza = xr.DataArray(dz, dims=[COORD_Z_CENTER])
    phis = dz_and_top_to_phis(top, dza)
    np.testing.assert_allclose(phis.values / GRAVITY, top + np.sum(dz))
