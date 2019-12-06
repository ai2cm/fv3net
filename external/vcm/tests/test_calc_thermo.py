import numpy as np
import xarray as xr

from vcm.calc.thermo import VERTICAL_DIM, absolute_pressure_interface


def test_absolute_pressure_interface():
    dp = xr.DataArray(np.arange(1, 10), dims=[VERTICAL_DIM])
    pabs = absolute_pressure_interface(dp)
    pabs = xr.DataArray(pabs)
    xr.testing.assert_allclose(pabs.diff(VERTICAL_DIM), dp)
