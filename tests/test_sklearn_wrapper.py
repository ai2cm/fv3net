import numpy as np
import pytest
import xarray as xr

from fv3net.regression.sklearn.wrapper import SklearnWrapper, _flatten


@pytest.fixture
def test_model_wrapper():
    from sklearn.ensemble import RandomForestRegressor

    regressor = RandomForestRegressor()
    model_wrapper = SklearnWrapper(regressor)
    target_means = np.array([1.0, 2.0, -1.0])
    target_stddevs = np.array([0.5, 1.0, 2.0])
    model_wrapper.save_normalization_data(target_means, target_stddevs)
    return model_wrapper


def test__flatten():
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a})

    ans = _flatten(ds, sample_dim)
    assert ans.shape == (nz, 2 * nx * ny)


def test__flatten_1d_input():
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a.isel(x=0, y=0)})

    ans = _flatten(ds, sample_dim)
    assert ans.shape == (nz, nx * ny + 1)


def test__flatten_same_order():
    nx, ny = 10, 4
    x = xr.DataArray(np.arange(nx * ny).reshape((nx, ny)), dims=["sample", "feature"])

    ds = xr.Dataset({"a": x, "b": x.T})
    sample_dim = "sample"
    a = _flatten(ds[["a"]], sample_dim)
    b = _flatten(ds[["b"]], sample_dim)

    np.testing.assert_allclose(a, b)


def test__norm_outputs(test_model_wrapper):
    physical_target_values = np.array([2.0, 4.0, -2])
    normed_target_values = test_model_wrapper.norm_outputs(physical_target_values)
    assert np.array_equal(normed_target_values, [2.0, 2.0, -0.5])


def test__unnorm_outputs(test_model_wrapper):
    normed_target_values = [2.0, 2.0, -0.5]
    physical_target_values = test_model_wrapper.unnorm_outputs(normed_target_values)
    assert np.array_equal(physical_target_values, [2.0, 4.0, -2])
