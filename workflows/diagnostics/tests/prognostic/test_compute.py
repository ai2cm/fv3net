import numpy as np
import xarray as xr

from fv3net.diagnostics.prognostic_run.compute import itcz_edges


def test_itcz_edges():
    lat = np.arange(-85, 90, 10)
    psi = xr.DataArray(np.linspace(-20, 40, len(lat)), coords={"latitude": lat})
    min_, max_ = itcz_edges(psi, lat="latitude")
    xr.testing.assert_identical(min_, xr.DataArray(-25.0, name="latitude"))
    xr.testing.assert_identical(max_, xr.DataArray(25.0, name="latitude"))
