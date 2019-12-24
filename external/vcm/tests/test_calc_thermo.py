import numpy as np
import pytest
import xarray as xr
from vcm.calc.thermo import (
    GRAVITY,
    pressure_at_interface,
    height_at_interface,
    _interface_to_midpoint,
    dz_and_top_to_phis,
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


def test__interface_to_midpoint():
    interface = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_OUTER])
    midpoint = _interface_to_midpoint(interface)
    xr.testing.assert_allclose(
        xr.DataArray(np.arange(1.5, 9), dims=[COORD_Z_CENTER]), midpoint
    )


def test_dz_and_top_to_phis():
    top = 10.0
    dz = np.arange(-4, -1)
    dza = xr.DataArray(dz, dims=[COORD_Z_CENTER])
    phis = dz_and_top_to_phis(top, dza)
    np.testing.assert_allclose(phis.values / GRAVITY, top + np.sum(dz))
