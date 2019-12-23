import numpy as np
import xarray as xr
from vcm.calc.thermo import (
    GRAVITY,
    pressure_at_interface,
    height_at_interface,
    _interface_to_midpoint,
    dz_and_top_to_phis,
)
from vcm.cubedsphere.constants import COORD_Z_CENTER


def test_pressure_on_interface():
    delp = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_CENTER])
    pabs = pressure_at_interface(delp)
    pabs = xr.DataArray(pabs)
    xr.testing.assert_allclose(pabs.diff(COORD_Z_CENTER), delp)


def test_height_on_interface():
    dz = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_CENTER])
    phis = xr.DataArray(0)
    zabs = height_at_interface(dz, phis)
    zabs = xr.DataArray(zabs)
    xr.testing.assert_allclose(zabs.diff(COORD_Z_CENTER), dz)


def test__interface_to_midpoint():
    interface = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_CENTER])
    midpoint = _interface_to_midpoint(interface, dim=COORD_Z_CENTER)
    xr.testing.assert_allclose(
        xr.DataArray(np.arange(1.5, 9), dims=[COORD_Z_CENTER]), midpoint
    )


def test_dz_and_top_to_phis():
    top = 10.0
    dz = np.arange(-4, -1)
    dza = xr.DataArray(dz, dims=[COORD_Z_CENTER])
    phis = dz_and_top_to_phis(top, dza)
    np.testing.assert_allclose(phis.values / GRAVITY, top + np.sum(dz))
