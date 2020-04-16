import xarray as xr
import pytest
import numpy as np

from fv3net.regression.sklearn import train
from fv3net.regression.dataset_handler import _shuffled


def test_train_save_output_succeeds(tmpdir):
    model = object()
    config = {"a": 1}
    url = str(tmpdir)
    train.save_output(url, model, config)


def _dataset(sample_dim):
    m, n = 10, 2
    x = "x"
    sample = sample_dim
    return xr.Dataset(
        {"a": ([sample, x], np.ones((m, n))), "b": ([sample], np.ones((m))),},
        coords={x: np.arange(n), sample_dim: np.arange(m)},
    )


def test__shuffled():
    dataset = _dataset("sample")
    dataset.isel(sample=1)
    _shuffled(dataset, "sample", 1)


def test__shuffled_dask():
    dataset = _dataset("sample").chunk()
    _shuffled(dataset, "sample", 1)

