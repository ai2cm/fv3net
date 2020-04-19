import xarray as xr
import numpy as np

from fv3net.regression.sklearn import train
from fv3net.regression.dataset_handler import _shuffled, _validate_stack_dims

import pytest


def test_train_save_model_succeeds(tmpdir):
    model = object()
    url = str(tmpdir)
    filename = "filename.pkl"
    train.save_model(url, model, filename)


def _dataset(sample_dim):
    m, n = 10, 2
    x = "x"
    sample = sample_dim
    return xr.Dataset(
        {"a": ([sample, x], np.ones((m, n))), "b": ([sample], np.ones((m)))},
        coords={x: np.arange(n), sample_dim: np.arange(m)},
    )


def test__shuffled():
    dataset = _dataset("sample")
    dataset.isel(sample=1)
    _shuffled(dataset, "sample", 1)


def test__shuffled_dask():
    dataset = _dataset("sample").chunk()
    _shuffled(dataset, "sample", 1)


def test__validate_stack_dims_ok():
    arr_2d = np.ones((10, 10))
    ds = xr.Dataset({"a": (["x", "y"], arr_2d), "b": (["x", "y"], arr_2d),})

    _validate_stack_dims(ds, ["x", "y"])


def test__validate_stack_dims_not_ok():
    arr_2d = np.ones((10, 10))
    arr_3d = np.ones((10, 10, 2))
    ds = xr.Dataset({"a": (["x", "y"], arr_2d), "b": (["x", "y", "z"], arr_3d),})

    with pytest.raises(ValueError):
        # don't allow us to broadcast over the "z" dimension
        _validate_stack_dims(ds, ["x", "y", "z"])

    _validate_stack_dims(ds, ["x", "y", "z"], allowed_broadcast_dims=["z"])
