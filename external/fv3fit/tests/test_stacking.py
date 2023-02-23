import numpy as np
import pytest
import xarray as xr
from fv3fit._shared.stacking import stack, SAMPLE_DIM_NAME, Z_DIM_NAMES

ds_unstacked = xr.Dataset(
    {"var": xr.DataArray(np.arange(0, 100).reshape(5, 20), dims=["z", "x"],)}
)
batches_unstacked = [ds_unstacked, ds_unstacked]


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
    s_dim = SAMPLE_DIM_NAME
    ds_train = stack(gridded_dataset, unstacked_dims=Z_DIM_NAMES)
    assert set(ds_train.dims) == {s_dim, "z"}
    assert len(ds_train["z"]) == len(gridded_dataset.z)
    assert ds_train["var"].dims[0] == s_dim


@pytest.mark.parametrize(
    "gridded_dataset", [(0, 1, 10, 10), (0, 10, 10, 10)], indirect=True,
)
def test_stack_dims_no_unstacked_dims(gridded_dataset):
    s_dim = SAMPLE_DIM_NAME
    ds_train = stack(gridded_dataset, unstacked_dims=["x", "y", "z"])
    assert set(ds_train.dims) == {s_dim, "x", "y", "z"}
    assert len(ds_train["z"]) == len(gridded_dataset.z)
    assert list(ds_train["var"].dims) == [s_dim, "x", "y", "z"]
    assert ds_train["var"].shape[0] == 1


def test_multiple_unstacked_dims():
    na, nb, nc, nd = 2, 3, 4, 5
    ds = xr.Dataset(
        data_vars={
            "var1": xr.DataArray(
                np.zeros([na, nb, nc, nd]), dims=["a", "b", "c", "d"],
            ),
            "var2": xr.DataArray(np.zeros([na, nb, nc]), dims=["a", "b", "c"],),
        }
    )
    unstacked_dims = ["c", "d"]
    expected = xr.Dataset(
        data_vars={
            "var1": xr.DataArray(
                np.zeros([na * nb, nc, nd]), dims=[SAMPLE_DIM_NAME, "c", "d"],
            ),
            "var2": xr.DataArray(np.zeros([na * nb, nc]), dims=[SAMPLE_DIM_NAME, "c"],),
        }
    )
    result = stack(ds=ds, unstacked_dims=unstacked_dims)
    xr.testing.assert_identical(result.drop(result.coords.keys()), expected)


@pytest.mark.parametrize("dims", [("time", "x", "y", "z"), ("time", "z", "y", "x")])
def test_multiple_unstacked_dims_are_alphabetically_ordered(dims):
    nt, nx, ny, nz = 2, 12, 12, 15
    ds = xr.Dataset(
        data_vars={"var1": xr.DataArray(np.zeros([nt, nx, ny, nz]), dims=dims,)}
    )
    unstacked_dims = ["x", "y", "z"]
    result = stack(ds=ds, unstacked_dims=unstacked_dims)
    assert list(result["var1"].dims) == [SAMPLE_DIM_NAME, "x", "y", "z"]
