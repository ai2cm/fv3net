import numpy as np
import pytest
import xarray as xr

from fv3net.regression.sklearn.wrapper import TargetTransformer, Packer


@pytest.fixture
def test_transformer():
    target_means = np.array([1.0, 2.0, -1.0])
    target_stddevs = np.array([0.5, 1.0, 2.0])
    transformer = TargetTransformer(target_means, target_stddevs)
    return transformer


@pytest.fixture
def test_packer():
    packer = Packer(["T", "sphum"], ["Q1", "Q2"], "z")
    return packer


def test_flatten(test_packer):
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a})

    ans = test_packer.flatten(ds, sample_dim)
    assert ans.shape == (nz, 2 * nx * ny)


def test_flatten_1d_input(test_packer):
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a.isel(x=0, y=0)})

    ans = test_packer.flatten(ds, sample_dim)
    assert ans.shape == (nz, nx * ny + 1)


def test_flatten_same_order(test_packer):
    nx, ny = 10, 4
    x = xr.DataArray(np.arange(nx * ny).reshape((nx, ny)), dims=["sample", "feature"])

    ds = xr.Dataset({"a": x, "b": x.T})
    sample_dim = "sample"
    a = test_packer.flatten(ds[["a"]], sample_dim)
    b = test_packer.flatten(ds[["b"]], sample_dim)

    np.testing.assert_allclose(a, b)


def test_transform(test_transformer):
    physical_target_values = np.array([2.0, 4.0, -2])
    normed_target_values = test_transformer.transform(physical_target_values)
    assert pytest.approx(normed_target_values) == [2.0, 2.0, -0.5]


def test_inverse_transform_outputs(test_transformer):
    normed_target_values = [2.0, 2.0, -0.5]
    physical_target_values = test_transformer.inverse_transform(normed_target_values)
    assert pytest.approx(physical_target_values) == [2.0, 4.0, -2]
