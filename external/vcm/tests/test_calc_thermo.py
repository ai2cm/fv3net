import numpy as np
import xarray as xr

from vcm.calc.thermo import (
    gravity,
    VERTICAL_DIM,
    pressure_at_interface,
    height_at_interface,
    _interface_to_midpoint,
    dz_and_top_to_phis,
)


def test_pressure_on_interface():
    delp = xr.DataArray(np.arange(1, 10), dims=[VERTICAL_DIM])
    pabs = pressure_at_interface(delp)
    pabs = xr.DataArray(pabs)
    xr.testing.assert_allclose(pabs.diff(VERTICAL_DIM), delp)


def test_height_on_interface():
    dz = xr.DataArray(np.arange(1, 10), dims=[VERTICAL_DIM])
    phis = xr.DataArray(0)
    zabs = height_at_interface(dz, phis)
    zabs = xr.DataArray(zabs)
    xr.testing.assert_allclose(zabs.diff(VERTICAL_DIM), dz)


def test__interface_to_center():
    interface = xr.DataArray(np.arange(1, 10), dims=[VERTICAL_DIM])
    center = _interface_to_midpoint(interface, dim=VERTICAL_DIM)
    xr.testing.assert_allclose(
        xr.DataArray(np.arange(1.5, 9), dims=[VERTICAL_DIM]), center
    )


def test_dz_and_top_to_phis():
    top = 10.0
    dz = np.arange(-4, -1)
    dza = xr.DataArray(dz, dims=[VERTICAL_DIM])
    phis = dz_and_top_to_phis(top, dza)
    np.testing.assert_allclose(phis.values / gravity, top + np.sum(dz))
