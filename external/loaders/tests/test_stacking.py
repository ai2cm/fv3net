import loaders
import xarray as xr
import numpy as np
from loaders._utils import SAMPLE_DIM_NAME
import pytest


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
    result = loaders.stack(ds=ds, unstacked_dims=unstacked_dims)
    xr.testing.assert_identical(result.drop(result.coords.keys()), expected)


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
    ds_train = loaders.stack(["z"], gridded_dataset)
    assert set(ds_train.dims) == {s_dim, "z"}
    assert len(ds_train["z"]) == len(gridded_dataset.z)
    assert ds_train["var"].dims[0] == s_dim
