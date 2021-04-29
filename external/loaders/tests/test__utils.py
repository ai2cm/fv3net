import numpy as np
import pytest
import xarray as xr
from loaders._utils import (
    shuffled,
    _ensure_sample_first,
    _get_chunk_indices,
    _get_z_dim,
    _group_by_z_dim,
    check_empty,
    stack_non_vertical,
    preserve_samples_per_batch,
    nonderived_variables,
    subsample,
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
    ds_train = stack_non_vertical(gridded_dataset, sample_dim_name=s_dim)
    assert set(ds_train.dims) == {s_dim, "z"}
    assert len(ds_train["z"]) == len(gridded_dataset.z)
    assert ds_train["var"].dims[0] == s_dim


@pytest.mark.parametrize(
    "gridded_dataset, expected_error",
    [((0, 2, 10, 10), None), ((10, 2, 10, 10), None), ((110, 2, 10, 10), ValueError)],
    indirect=["gridded_dataset"],
)
def test_check_empty(gridded_dataset, expected_error):
    s_dim = "sample"
    ds_grid = stack_non_vertical(gridded_dataset, sample_dim_name=s_dim)
    no_nan = ds_grid.dropna(s_dim)
    if expected_error is not None:
        with pytest.raises(expected_error):
            check_empty(no_nan)
    else:
        check_empty(no_nan)


@pytest.mark.parametrize(
    "gridded_dataset", [(0, 2, 10, 10)], indirect=True,
)
def test_subsample_dim(gridded_dataset):
    s_dim = "sample"
    ds_train = stack_non_vertical(gridded_dataset, sample_dim_name=s_dim)
    n = 10
    subsample_func = subsample(n, np.random.RandomState(0))
    subsampled = subsample_func(ds_train, dim=s_dim)
    assert subsampled.sizes[s_dim] == n
    for dim in subsampled.sizes:
        if dim != s_dim:
            assert ds_train.sizes[dim] == subsampled.sizes[dim]

    with pytest.raises(KeyError):
        subsample_func(ds_train, dim="not_a_dim")


@pytest.mark.parametrize(
    "gridded_dataset", [(0, 2, 10, 10)], indirect=True,
)
def test_preserve_samples_per_batch(gridded_dataset):
    num_multiple = 4
    d_dim = "dataset"
    s_dim = "sample"

    multi_ds = xr.concat([gridded_dataset] * num_multiple, dim=d_dim)
    stacked = stack_non_vertical(multi_ds, sample_dim_name=s_dim)
    thinned = preserve_samples_per_batch(stacked, dataset_dim_name=d_dim)
    orig_stacked = stack_non_vertical(gridded_dataset, sample_dim_name=s_dim)

    samples_per_batch_diff = thinned.sizes[s_dim] - orig_stacked.sizes[s_dim]
    # Thinning operation can be at most the size of dataset dim extra depending
    # on the index stride
    assert samples_per_batch_diff <= num_multiple


@pytest.mark.parametrize(
    "gridded_dataset", [(0, 2, 10, 10)], indirect=True,
)
def test_preserve_samples_per_batch_not_multi(gridded_dataset):
    s_dim = "sample"
    stacked = stack_non_vertical(gridded_dataset, sample_dim_name=s_dim)
    result = preserve_samples_per_batch(stacked)
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


def test_shuffled():
    dataset = _stacked_dataset("sample")
    dataset.isel(sample=1)
    shuffled(np.random.RandomState(1), dataset, dim="sample")


def test_shuffled_dask():
    dataset = _stacked_dataset("sample").chunk()
    shuffled(np.random.RandomState(1), dataset, dim="sample")


def test__ensure_sample_first():
    s_dim = "sample"
    ds = xr.Dataset(
        data_vars={
            "correct": xr.DataArray(np.ones((4, 6)), dims=[s_dim, "z"]),
            "incorrect": xr.DataArray(np.ones((6, 4)), dims=["z", s_dim]),
        }
    )

    sample_first = _ensure_sample_first(ds, sample_dim_name=s_dim)
    for da in sample_first.values():
        assert da.dims[0] == s_dim


def test__ensure_sample_first_error_on_3d():
    ds = xr.Dataset(
        data_vars={"var": xr.DataArray(np.ones((4, 6, 8)), dims=["x", "y", "z"]),}
    )
    with pytest.raises(ValueError):
        _ensure_sample_first(ds)


def test__get_z_dim():
    dims = ["x", "y", "z"]
    vert_dim = "z"

    result = _get_z_dim(dims, z_dim_names=[vert_dim])
    assert result == vert_dim

    result = _get_z_dim(dims[0:2], z_dim_names=[vert_dim])
    assert result is None


def test__get_z_dim_multiple_vert_dims():
    with pytest.raises(ValueError):
        _get_z_dim(["x", "z", "z_soil"], z_dim_names=["z", "z_soil"])


def test__group_by_z_dim():
    z_dims = ["z", "z_soil"]
    ds = xr.Dataset(
        data_vars={
            "vert_var": xr.DataArray(np.ones((4, 6)), dims=["x", "z"]),
            "2d_var": xr.DataArray(np.ones(4), dims=["x"]),
            "alt_vert_var": xr.DataArray(np.ones((4, 2)), dims=["x", "z_soil"]),
        }
    )

    groups = _group_by_z_dim(ds, z_dim_names=z_dims)
    for group_ds in groups.values():
        assert len(group_ds.data_vars) == 1
    assert "vert_var" in groups["z"].data_vars
    assert "alt_vert_var" in groups["z_soil"].data_vars
    assert "2d_var" in groups["no_vertical"].data_vars
