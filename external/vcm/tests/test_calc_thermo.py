import numpy as np
import xarray as xr

from vcm.calc.thermo import (
    gravity,
    VERTICAL_DIM,
    pressure_on_interface,
    height_on_interface,
    interface_to_center,
    dz_and_top_to_phis,
)


def test_pressure_on_interface():
    delp = xr.DataArray(np.arange(1, 10), dims=[VERTICAL_DIM])
    pabs = pressure_on_interface(delp)
    pabs = xr.DataArray(pabs)
    xr.testing.assert_allclose(pabs.diff(VERTICAL_DIM), delp)


def test_height_on_interface():
    dz = xr.DataArray(np.arange(1, 10), dims=[VERTICAL_DIM])
    phis = xr.DataArray(0)
    zabs = height_on_interface(dz, phis)
    zabs = xr.DataArray(zabs)
    xr.testing.assert_allclose(zabs.diff(VERTICAL_DIM), dz)


def test_interface_to_center():
    interface = xr.DataArray(np.arange(1, 10), dims=[VERTICAL_DIM])
    center = interface_to_center(interface, dim=VERTICAL_DIM)
    xr.testing.assert_allclose(np.arange(1.5, 9), center)


def test_dz_and_top_to_phis():
    top = 10.0
    dz = np.arange(-4, -1)
    dza = xr.DataArray(dz, dims=[VERTICAL_DIM])
    phis = dz_and_top_to_phis(top, dza)
    xr.testing.assert_allclose(phis / gravity, top + np.sum(dz))
