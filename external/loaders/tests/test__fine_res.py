import xarray as xr
import numpy as np
import pytest
from loaders.mappers._fine_res import _extend_lower


@pytest.mark.parametrize(
    ["nx", "nz", "vertical_dim", "error"],
    (
        pytest.param(1, 5, "z", False, id="base"),
        pytest.param(5, 1, "x", False, id="different_dim"),
        pytest.param(5, 1, "z", True, id="scalar_dim"),
    ),
)
def test__extend_lower(nz, nx, vertical_dim, error):
    f = xr.DataArray(np.arange(nx * nz).reshape((nx, nz)), dims=["x", "z"])
    f.attrs = {"long_name": "some source term", "description": "some source term"}
    ds = xr.Dataset({"f": f})
    if error:
        with pytest.raises(ValueError):
            extended = _extend_lower(ds.f, vertical_dim)
    else:
        extended = _extend_lower(ds.f, vertical_dim)
        xr.testing.assert_allclose(
            extended.isel({vertical_dim: -1}), extended.isel({vertical_dim: -2})
        )
