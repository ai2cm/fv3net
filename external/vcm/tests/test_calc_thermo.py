import numpy as np
import xarray as xr

from vcm.calc.thermo import (
    VERTICAL_DIM,
    pressure_on_interface,
    height_on_interface,
)


def test_absolute_pressure_interface():
    delp = xr.DataArray(np.arange(1, 10), dims=[VERTICAL_DIM])
    pabs = pressure_on_interface(delp)
    pabs = xr.DataArray(pabs)
    xr.testing.assert_allclose(pabs.diff(VERTICAL_DIM), delp)


def test_absolute_height_interface():
    dz = xr.DataArray(np.arange(1, 10), dims=[VERTICAL_DIM])
    phis = xr.DataArray(0)
    zabs = height_on_interface(dz, phis)
    zabs = xr.DataArray(zabs)
    xr.testing.assert_allclose(zabs.diff(VERTICAL_DIM), dz)