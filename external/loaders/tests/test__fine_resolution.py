import xarray as xr
import numpy as np
import pytest
from loaders.mappers._fine_res import _extend_lower


@pytest.mark.parametrize(
    ["nx", "nz", "vertical_dim", "n_levels", "error"],
    (
        pytest.param(1, 5, "z", 2, False, id="base"),
        pytest.param(5, 1, "x", 2, False, id="different_dim"),
        pytest.param(1, 5, "z", 3, False, id="different_n_levels"),
        pytest.param(5, 1, "z", 2, True, id="scalar_dim"),
    ),
)
def test__extend_lower(nz, nx, vertical_dim, n_levels, error):
    f = xr.DataArray(np.arange(nx * nz).reshape((nx, nz)), dims=["x", "z"])
    f.attrs = {"long_name": "some source term", "description": "some source term"}
    ds = xr.Dataset({"f": f})
    if error:
        with pytest.raises(ValueError):
            extended = _extend_lower(ds.f, vertical_dim, n_levels)
    else:
        extended = _extend_lower(ds.f, vertical_dim, n_levels)
        expected = extended.isel({vertical_dim: -(n_levels + 1)})
        [
            xr.testing.assert_allclose(
                extended.isel({vertical_dim: -(i + 1)}), expected
            )
            for i in range(n_levels)
        ]
