import numpy as np
import xarray as xr
import pytest
from vcm.limit import DatasetQuantileLimiter

EPS = 1.0e-3


def _get_arr(upper_limit, lower_limit):
    scale = 1.0 / (1.0 - upper_limit + lower_limit)
    xscaling = np.arange(0.0, (scale + EPS))[:, np.newaxis]
    zscaling = np.ones((1, int(scale) + 1))
    return xscaling * zscaling


def get_dataset(upper_limit, lower_limit):
    """Dataset of two variables that each vary in x but not z, scaled such that the
        first and last column in x will be outside the lower_limit and upper_limit,
        respectively
    
    E.g., for upper limit of 0.875 and lower limit of 0.125, each z level will be
    [0.0, 1.0, 2.0, 3.0, 4.0]
    and its "limited" equivalent would be
    [0.5, 1.0, 2.0, 3.0, 3.5]
    """
    arr = _get_arr(upper_limit, lower_limit)
    da = xr.DataArray(arr, dims=["x", "z"])
    return xr.Dataset({"Q1": da, "Q2": da})


def get_limits(upper_expected, lower_expected, upper_limit, lower_limit, feature_dims):
    scale = int(1.0 / (1.0 - upper_limit + lower_limit)) + 1
    upper_da = xr.DataArray(np.broadcast_to(upper_expected, scale), dims=feature_dims)
    lower_da = xr.DataArray(np.broadcast_to(lower_expected, scale), dims=feature_dims)
    return (
        xr.Dataset({"Q1": upper_da, "Q2": upper_da}),
        xr.Dataset({"Q1": lower_da, "Q2": lower_da}),
    )


@pytest.mark.parametrize(
    ["upper_limit", "lower_limit", "feature_dims", "upper_expected", "lower_expected"],
    [
        pytest.param(0.875, 0.125, ["z"], 3.5, 0.5, id="alpha=0.25"),
        pytest.param(0.9, 0.1, ["z"], 4.5, 0.5, id="alpha=0.2"),
        pytest.param(
            0.875,
            0.125,
            ["x"],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            id="feature_dim_x",
        ),
    ],
)
def test_limiter_fit(
    upper_limit, lower_limit, feature_dims, upper_expected, lower_expected
):
    ds = get_dataset(upper_limit, lower_limit)
    limiter = DatasetQuantileLimiter(upper_limit, lower_limit)
    limiter.fit(ds, feature_dims=feature_dims)
    upper, lower = get_limits(
        upper_expected, lower_expected, upper_limit, lower_limit, feature_dims
    )
    xr.testing.assert_allclose(limiter._upper, upper)
    xr.testing.assert_allclose(limiter._lower, lower)


@pytest.mark.parametrize(
    "limit_only", [pytest.param(None, id="all"), pytest.param(["Q1"], id="Q1_only")]
)
def test_limiter_transform_limit_only(limit_only):
    upper_limit, lower_limit = 0.875, 0.125
    ds = get_dataset(upper_limit, lower_limit)
    limiter = DatasetQuantileLimiter(upper_limit, lower_limit, limit_only=limit_only)
    limiter._upper, limiter._lower = get_limits(
        3.5, 0.5, upper_limit, lower_limit, ["z"]
    )
    limited = limiter.transform(ds, deepcopy=True)
    expected_ds = ds.copy(deep=True)
    limit_only = limit_only if limit_only is not None else ds.data_vars
    for var in limit_only:
        expected_ds[var].loc[{"x": 0}] = 0.5
        expected_ds[var].loc[{"x": -1}] = 3.5
    xr.testing.assert_allclose(limited, expected_ds)
