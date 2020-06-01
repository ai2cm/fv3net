import numpy as np
import pytest
import xarray as xr
from fv3net.regression.loaders._transform import (
    shuffled,
    _get_chunk_indices,
    stack_dropnan_shuffle,
)


@pytest.fixture
def test_gridded_dataset(request):
    zdim, num_nans = request.param
    coords = {"z": range(zdim), "y": range(10), "x": range(10)}
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
    "test_gridded_dataset", [(2, 0), (2, 10), (2, 110), (2, 200), (1, 0)], indirect=True
)
def test_stack_dropnan_shuffle(test_gridded_dataset):
    ds_grid = test_gridded_dataset
    nan_mask_2d = ~np.isnan(
        ds_grid["var"].sum("z", skipna=False)
    )  # mask if any z coord has nan
    zdim = ds_grid.sizes["z"]
    num_finite_samples = np.count_nonzero(nan_mask_2d)
    flattened = ds_grid["var"].where(nan_mask_2d).values.flatten()
    finite_samples = flattened[~np.isnan(flattened)]
    rs = np.random.RandomState(seed=0)

    if num_finite_samples == 0:
        with pytest.raises(ValueError):
            ds_train = stack_dropnan_shuffle(
                init_time_dim_name="initial_time", random_state=rs, ds=ds_grid
            )
    else:
        ds_train = stack_dropnan_shuffle(
            init_time_dim_name="initial_time", random_state=rs, ds=ds_grid
        )
        assert set(ds_train.dims) == {"sample", "z"}
        assert len(ds_train["sample"]) == num_finite_samples
        assert len(ds_train["z"]) == zdim
        assert set(ds_train["var"].values.flatten()) == set(finite_samples)


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
