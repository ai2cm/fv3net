import numpy as np
import pytest
import xarray as xr
from fv3net.regression.loaders._transform import (
    shuffled,
    _get_chunk_indices,
    transform_train_data,
)


@pytest.fixture
def test_gridded_dataset():
    coords = {"z": range(2), "y": range(10), "x": range(10)}
    var = xr.DataArray(
        [[range(10) for i in range(10)] for j in range(2)],
        dims=["z", "y", "x"],
        coords=coords,
    )
    return xr.Dataset({"var": var})


def test_transform_train_data(test_gridded_dataset):
    ds = test_gridded_dataset
    da = ds["var"]
    ds["var"] = da.where(da != 0)  # assign 10/100 nan values
    ds_train = transform_train_data(
        init_time_dim_name="initial_time", random_seed=0, ds=ds
    )
    assert set(ds_train.dims) == {"sample", "z"}
    assert len(ds_train["sample"]) == 90
    assert len(ds_train["z"]) == 2


def test__get_chunk_indices():
    chunks = (2, 3)
    expected = [[0, 1], [2, 3, 4]]
    ans = _get_chunk_indices(chunks)
    assert ans == expected


def _stacked_dataset(sample_dim):
    m, n = 10, 2
    x = "x"
    sample = sample_dim
    return xr.Dataset(
        {"a": ([sample, x], np.ones((m, n))), "b": ([sample], np.ones((m)))},
        coords={x: np.arange(n), sample_dim: np.arange(m)},
    )


def test_shuffled():
    dataset = _stacked_dataset("sample")
    dataset.isel(sample=1)
    shuffled(dataset, "sample", np.random.RandomState(1))


def test_shuffled_dask():
    dataset = _stacked_dataset("sample").chunk()
    shuffled(dataset, "sample", np.random.RandomState(1))
