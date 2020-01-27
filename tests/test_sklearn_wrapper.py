import numpy as np
from sklearn.linear_model import LinearRegression
import pytest
import xarray as xr

from fv3net.regression.sklearn.wrapper import RegressorEnsemble, _pack


def test_flatten():
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a})

    ans = _pack(ds, sample_dim)[0]
    assert ans.shape == (nz, 2 * nx * ny)


def test_flatten_1d_input():
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a.isel(x=0, y=0)})

    ans = _pack(ds, sample_dim)[0]
    assert ans.shape == (nz, nx * ny + 1)


def test_flatten_same_order():
    nx, ny = 10, 4
    x = xr.DataArray(np.arange(nx * ny).reshape((nx, ny)), dims=["sample", "feature"])

    ds = xr.Dataset({"a": x, "b": x.T})
    sample_dim = "sample"
    a = _pack(ds[["a"]], sample_dim)[0]
    b = _pack(ds[["b"]], sample_dim)[0]

    np.testing.assert_allclose(a, b)


@pytest.fixture
def test_regressor_ensemble():
    base_regressor = LinearRegression()
    ensemble_regressor = RegressorEnsemble(base_regressor)
    num_batches = 3
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    for i in range(num_batches):
        ensemble_regressor.fit(X, y)
    return ensemble_regressor


def test_ensemble_fit(test_regressor_ensemble):
    regressor_ensemble = test_regressor_ensemble
    assert regressor_ensemble.n_estimators == 3
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    regressor_ensemble.fit(X, y)
    # test that .fit appends a new regressor
    assert regressor_ensemble.n_estimators == 4
    # test that new regressors are actually fit and not empty base regressor
    assert len(regressor_ensemble.regressors[-1].coef_) > 0
