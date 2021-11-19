import numpy as np
import xarray as xr
from loaders.mappers._hybrid import _convergence, _extend_lower
import pytest


def test__convergence_constant():
    nz = 5
    delp = np.ones(nz).reshape((1, 1, nz))

    expected = np.array([0, 0, 0, 0, 0]).reshape((1, 1, nz))

    ans = _convergence(delp, delp)
    np.testing.assert_almost_equal(ans, expected)


def test__convergence_linear():
    nz = 5
    f = np.arange(nz).reshape((1, 1, nz))
    delp = np.ones(nz).reshape((1, 1, nz))

    expected = np.array([-1, -1, -1, -1, -1]).reshape((1, 1, nz))

    ans = _convergence(f, delp)
    np.testing.assert_almost_equal(ans, expected)


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
