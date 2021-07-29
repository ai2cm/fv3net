import numpy as np
import pytest
import xarray as xr
from typing import Sequence
from fv3fit._shared.stacking import (
    _shuffled,
    _get_chunk_indices,
    _check_empty,
    stack_non_vertical,
    _preserve_samples_per_batch,
    StackedBatches,
)

ds_unstacked = xr.Dataset(
    {"var": xr.DataArray(np.arange(0, 100).reshape(5, 20), dims=["z", "x"],)}
)
batches_unstacked = [ds_unstacked, ds_unstacked]


def test_StackedBatches_get_index_stack():
    stacked_batches = StackedBatches(batches_unstacked, np.random.RandomState(0))
    stacked_batch = stacked_batches[0]
    assert stacked_batch["var"].dims == ("sample", "z")
    assert len(stacked_batch.sample) == 20


def test_StackedBatches_get_slice_stack():
    stacked_batches = StackedBatches(batches_unstacked, np.random.RandomState(0))
    stacked_batch_sequence = stacked_batches[slice(0, None)]
    assert isinstance(stacked_batch_sequence, Sequence)
    assert len(stacked_batch_sequence) == 2
    for ds in stacked_batch_sequence:
        assert ds["var"].dims == ("sample", "z")
        assert len(ds.sample) == 20


@pytest.fixture
def gridded_dataset(request):
    num_nans, zdim, ydim, xdim = request.param
    coords = {"z": range(zdim), "y": range(ydim), "x": range(xdim)}
    # unique values for ease of set comparison in test
    var = xr.DataArray(
        [
            [[(100 * k) + (10 * j) + i for i in range(10)] for j in range(10)]
            for k in range(zdim)
        ],
        dims=["z", "y", "x"],
        coords=coords,
    )
    var = var.where(var >= num_nans)  # assign nan values
    return xr.Dataset({"var": var})


@pytest.mark.parametrize(
    "gridded_dataset", [(0, 1, 10, 10), (0, 10, 10, 10)], indirect=True,
)
def test_stack_dims(gridded_dataset):
    s_dim = "sample"
    ds_train = stack_non_vertical(gridded_dataset)
    assert set(ds_train.dims) == {s_dim, "z"}
    assert len(ds_train["z"]) == len(gridded_dataset.z)
    assert ds_train["var"].dims[0] == s_dim


@pytest.mark.parametrize(
    "gridded_dataset, expected_error",
    [((0, 2, 10, 10), None), ((10, 2, 10, 10), None), ((110, 2, 10, 10), ValueError)],
    indirect=["gridded_dataset"],
)
def test__check_empty(gridded_dataset, expected_error):
    s_dim = "sample"
    ds_grid = stack_non_vertical(gridded_dataset)
    no_nan = ds_grid.dropna(s_dim)
    if expected_error is not None:
        with pytest.raises(expected_error):
            _check_empty(no_nan)
    else:
        _check_empty(no_nan)


@pytest.mark.parametrize(
    "gridded_dataset", [(0, 2, 10, 10)], indirect=True,
)
def test__preserve_samples_per_batch(gridded_dataset):
    num_multiple = 4
    d_dim = "dataset"
    s_dim = "sample"

    multi_ds = xr.concat([gridded_dataset] * num_multiple, dim=d_dim)
    stacked = stack_non_vertical(multi_ds)
    thinned = _preserve_samples_per_batch(stacked)
    orig_stacked = stack_non_vertical(gridded_dataset)

    samples_per_batch_diff = thinned.sizes[s_dim] - orig_stacked.sizes[s_dim]
    # Thinning operation can be at most the size of dataset dim extra depending
    # on the index stride
    assert samples_per_batch_diff <= num_multiple


@pytest.mark.parametrize(
    "gridded_dataset", [(0, 2, 10, 10)], indirect=True,
)
def test__preserve_samples_per_batch_not_multi(gridded_dataset):
    s_dim = "sample"
    stacked = stack_non_vertical(gridded_dataset)
    result = _preserve_samples_per_batch(stacked)
    assert result.sizes[s_dim] == stacked.sizes[s_dim]


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


def test__shuffled():
    dataset = _stacked_dataset("sample")
    dataset.isel(sample=1)
    _shuffled(np.random.RandomState(1), dataset)


def test__shuffled_dask():
    dataset = _stacked_dataset("sample").chunk()
    _shuffled(np.random.RandomState(1), dataset)
