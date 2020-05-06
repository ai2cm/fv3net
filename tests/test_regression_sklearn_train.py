import xarray as xr
import numpy as np

from fv3net.regression.sklearn import train
from fv3net.regression.dataset_handler import _shuffled


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
    _shuffled(dataset, "sample", np.random.RandomState(1))


def test__shuffled_dask():
    dataset = _dataset("sample").chunk()
    _shuffled(dataset, "sample", np.random.RandomState(1))
