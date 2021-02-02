import numpy as np
import pytest
import xarray as xr
from loaders._utils import (
    shuffled,
    _get_chunk_indices,
    drop_nan,
    stack,
    preserve_samples_per_batch,
    nonderived_variables,
    _needs_grid_data,
)


@pytest.mark.parametrize(
    "requested, available, nonderived",
    (
        [["dQ1", "dQ2"], ["dQ1", "dQ2"], ["dQ1", "dQ2"]],
        [
            ["dQ1", "dQ2", "dQu", "dQv", "cos_zenith_angle"],
            ["dQ1", "dQ2"],
            ["dQ1", "dQ2", "dQxwind", "dQywind"],
        ],
        [
            ["dQ1", "dQ2", "dQu", "dQv", "cos_zenith_angle"],
            ["dQ1", "dQ2", "dQu", "dQv"],
            ["dQ1", "dQ2", "dQu", "dQv"],
        ],
    ),
)
def test_nonderived_variable_names(requested, available, nonderived):
    assert set(nonderived_variables(requested, available)) == set(nonderived)


@pytest.fixture
def test_gridded_dataset(request):
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
    "test_gridded_dataset", [(0, 1, 10, 10), (0, 10, 10, 10)], indirect=True,
)
def test_stack_dims(test_gridded_dataset):
    ds_train = stack(test_gridded_dataset)
    assert set(ds_train.dims) == {"sample", "z"}
    assert len(ds_train["z"]) == len(test_gridded_dataset.z)
    assert ds_train["var"].dims[0] == "sample"


@pytest.mark.parametrize(
    "test_gridded_dataset, num_finite_samples",
    [((0, 2, 10, 10), 100), ((10, 2, 10, 10), 90), ((110, 2, 10, 10), 0)],
    indirect=["test_gridded_dataset"],
)
def test_dropnan_samples(test_gridded_dataset, num_finite_samples):
    ds_grid = stack(test_gridded_dataset)
    nan_mask_2d = ~np.isnan(
        ds_grid["var"].sum("z", skipna=False)
    )  # mask if any z coord has nan
    flattened = ds_grid["var"].where(nan_mask_2d).values.flatten()
    finite_samples = flattened[~np.isnan(flattened)]

    if num_finite_samples == 0:
        with pytest.raises(ValueError):
            ds_train = drop_nan(ds_grid)
    else:
        ds_train = drop_nan(ds_grid)
        assert len(ds_train["sample"]) == num_finite_samples
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


@pytest.mark.parametrize(
    "requested, existing, needs_grid",
    [
        (["x0", "x1"], ["x0", "x1"], False),
        (["x0", "cos_zenith_angle"], ["x0", "x1"], True),
        (["dQu"], ["x0", "x1"], True),
        (["dQu"], ["x0", "dQu"], True),
    ],
)
def test__needs_grid_data(requested, existing, needs_grid):
    assert _needs_grid_data(requested, existing) == needs_grid
