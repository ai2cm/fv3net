import numpy as np
import xarray as xr
import pytest
from vcm.limit import DatasetQuantileLimiter

EPS = 1.0e-3


def _xscaling(alpha):
    """make data that varies along an x dim"""
    scale = 1.0 / alpha
    return np.arange(0.0, (scale + EPS))


def _get_arr(alpha):
    xscaling = _xscaling(alpha)[:, np.newaxis]
    zscaling = np.ones((1, int(1.0 / alpha) + 1))
    return xscaling * zscaling


def get_dataset(alpha):
    """dataset of two variables that vary in x but not z"""
    arr = _get_arr(alpha)
    da = xr.DataArray(arr, dims=["x", "z"])
    return xr.Dataset({"Q1": da, "Q2": da})


def get_limits(upper, lower, alpha, feature_dims):
    scale = int(1.0 / alpha) + 1
    upper_da = xr.DataArray(np.broadcast_to(upper, scale), dims=feature_dims)
    lower_da = xr.DataArray(np.broadcast_to(lower, scale), dims=feature_dims)
    return (
        xr.Dataset({"Q1": upper_da, "Q2": upper_da}),
        xr.Dataset({"Q1": lower_da, "Q2": lower_da}),
    )


@pytest.mark.parametrize(
    ["alpha", "feature_dims", "fit_indexers", "error", "upper", "lower"],
    [
        pytest.param(0.25, ["z"], None, False, 3.5, 0.5, id="alpha=0.25"),
        pytest.param(0.2, ["z"], None, False, 4.5, 0.5, id="alpha=0.2"),
        pytest.param(
            0.25,
            ["x"],
            None,
            False,
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            id="feature_dim_x",
        ),
        pytest.param(
            0.25, ["z"], {"x": slice(None, 3)}, False, 1.75, 0.25, id="fit_indexer_x"
        ),
        pytest.param(0.2, ["z"], {"z": 0}, True, None, None, id="invalid_indexer_dim"),
    ],
)
def test_limiter_fit(alpha, feature_dims, fit_indexers, error, upper, lower):
    ds = get_dataset(alpha)
    limiter = DatasetQuantileLimiter(alpha=alpha)
    if not error:
        limiter.fit(ds, feature_dims=feature_dims, fit_indexers=fit_indexers)
        upper, lower = get_limits(upper, lower, alpha, feature_dims)
        xr.testing.assert_allclose(limiter._upper, upper)
        xr.testing.assert_allclose(limiter._lower, lower)
    else:
        with pytest.raises(ValueError):
            limiter.fit(ds, feature_dims=feature_dims, fit_indexers=fit_indexers)


@pytest.mark.parametrize(
    "limit_only", [pytest.param(None, id="all"), pytest.param(["Q1"], id="Q1_only")]
)
def test_limiter_transform_limit_only(limit_only):
    alpha = 0.25
    ds = get_dataset(alpha)
    limiter = DatasetQuantileLimiter(alpha, limit_only=limit_only)
    limiter._upper, limiter._lower = get_limits(3.5, 0.5, alpha, ["z"])
    limited = limiter.transform(ds, deepcopy=True)
    expected_ds = ds.copy(deep=True)
    for var in limit_only or ds.data_vars:
        expected_ds[var].loc[{"x": 0}] = 0.5
        expected_ds[var].loc[{"x": -1}] = 3.5
    xr.testing.assert_allclose(limited, expected_ds)
