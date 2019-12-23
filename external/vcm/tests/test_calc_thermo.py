import numpy as np
import xarray as xr
from vcm.calc.thermo import (
    GRAVITY,
    pressure_at_interface,
    height_at_interface,
    _interface_to_midpoint,
    dz_and_top_to_phis,
)
from vcm.cubedsphere.constants import COORD_Z_CENTER, COORD_Z_OUTER


def test_pressure_on_interface():
    delp = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_CENTER])
    pressure = pressure_at_interface(delp)
    xr.testing.assert_allclose(
        pressure.diff(COORD_Z_OUTER), delp.rename({COORD_Z_CENTER: COORD_Z_OUTER})
    )


def test_height_on_interface():
    dz = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_CENTER])
    phis = xr.DataArray(0)
    height = height_at_interface(dz, phis)
    xr.testing.assert_allclose(
        height.diff(COORD_Z_OUTER), dz.rename({COORD_Z_CENTER: COORD_Z_OUTER})
    )


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
