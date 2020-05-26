from fv3net.regression.loaders._transform import _shuffled, _chunk_indices
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def test_dataset():
    m, n = 10, 2
    x = "x"
    sample = "sample"
    return xr.Dataset(
        {"a": ([sample, x], np.ones((m, n))), "b": ([sample], np.ones((m)))},
        coords={x: np.arange(n), sample: np.arange(m)},
    )


def test__shuffled(test_dataset):
    dataset = test_dataset.isel(sample=1)
    _shuffled(dataset, "sample", np.random.RandomState(1))


def test__shuffled_dask(test_dataset):
    dataset = test_dataset.chunk()
    _shuffled(dataset, "sample", np.random.RandomState(1))


def test__chunk_indices(test_dataset):
    chunks = (2, 3)
    expected = [[0, 1], [2, 3, 4]]
    ans = _chunk_indices(chunks)
    assert ans == expected
