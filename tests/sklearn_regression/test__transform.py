import numpy as np
import xarray as xr
from fv3net.regression.loaders._transform import shuffled, _get_chunk_indices


def test__get_chunk_indices():
    chunks = (2, 3)
    expected = [[0, 1], [2, 3, 4]]
    ans = _get_chunk_indices(chunks)
    assert ans == expected


def _dataset(sample_dim):
    m, n = 10, 2
    x = "x"
    sample = sample_dim
    return xr.Dataset(
        {"a": ([sample, x], np.ones((m, n))), "b": ([sample], np.ones((m)))},
        coords={x: np.arange(n), sample_dim: np.arange(m)},
    )


def test_shuffled():
    dataset = _dataset("sample")
    dataset.isel(sample=1)
    shuffled(dataset, "sample", np.random.RandomState(1))


def test_shuffled_dask():
    dataset = _dataset("sample").chunk()
    shuffled(dataset, "sample", np.random.RandomState(1))
