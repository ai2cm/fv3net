import numpy as np
import xarray as xr

from fv3net.diagnostics.prognostic_run.compute import itcz_edges


def test_itcz_edges():
    lat = np.arange(-85, 90, 5)
    psi = xr.DataArray(np.sin(90 / 20 * np.deg2rad(lat)), coords={"latitude": lat})
    min_, max_ = itcz_edges(psi, lat="latitude")
    xr.testing.assert_identical(min_, xr.DataArray(-20.0, name="latitude"))
    xr.testing.assert_identical(max_, xr.DataArray(20.0, name="latitude"))
