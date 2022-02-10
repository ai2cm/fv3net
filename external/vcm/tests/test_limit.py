import numpy as np
import xarray as xr
import pytest
from vcm.limit import DatasetQuantileLimiter

EPS = 1.0e-3


def _xscaling(alpha):
    scale = 1.0 / alpha
    return np.arange(-scale, (scale + EPS))


def _zscaling(alpha):
    scale = 1.0 / alpha
    return np.flip(np.arange(0.0, scale + EPS) / scale)


def _get_arr(alpha):
    xscaling = _xscaling(alpha)[:, np.newaxis]
    zscaling = _zscaling(alpha)[np.newaxis, :]
    return xscaling * zscaling

    """        else:
            upper = (scale - 0.5) * (np.arange(scale, 0.0 - EPS, -1.0) / scale)
            lower = 0.5 * (np.arange(scale, 0.0 - EPS, -1.0) / scale)
            expected_arr[-1, :] = upper
            expected_arr[0 : (int(scale) + 1), :] = lower"""


def get_dataset(alpha):
    arr = _get_arr(alpha)
    da = xr.DataArray(arr, dims=["x", "z"])
    return xr.Dataset({"Q1": da, "Q2": da})


def get_expected_arr(alpha, feature_dims, fit_indexers):
    arr = _get_arr(alpha)
    if fit_indexers is None:
        if feature_dims == ["z"]:
            upper_limit = np.quantile(_xscaling(alpha), 1.0 - alpha / 2.0)
            upper = upper_limit * _zscaling(alpha)
            lower = -upper
            arr[-1, :] = upper
            arr[0, :] = lower
        elif feature_dims == ["x"]:
            upper_limit = np.quantile(_zscaling(alpha), 1.0 - alpha / 2.0)
            upper = upper_limit * _xscaling(alpha)
            lower_limit = np.quantile(_zscaling(alpha), alpha / 2.0)
            lower = lower_limit * _xscaling(alpha)
            arr[:, -1] = lower
            arr[:, 0] = upper
        else:
            raise ValueError("Only 'x' and 'z' are tested as feature_dims.")
    elif "x" in fit_indexers:
        upper_limit = np.quantile(
            _xscaling(alpha)[fit_indexers["x"]], 1.0 - alpha / 2.0
        )
        upper = upper_limit * _zscaling(alpha)
        lower_limit = np.quantile(_xscaling(alpha)[fit_indexers["x"]], alpha / 2.0)
        lower = lower_limit * _zscaling(alpha)
        arr[-1, :] = upper
        midpoint = int(1.0 / alpha) + 1
        arr[0:midpoint, :] = lower
    return arr


@pytest.mark.parametrize(
    ["alpha", "limit_only", "feature_dims", "fit_indexers", "error"],
    [
        pytest.param(0.1, None, ["z"], None, False, id="alpha=0.1"),
        pytest.param(0.2, None, ["z"], None, False, id="alpha=0.2"),
        pytest.param(0.1, ["Q1"], ["z"], None, False, id="limit_only_Q1"),
        pytest.param(0.1, None, ["x"], None, False, id="feature_dim_x"),
        pytest.param(
            0.1, None, ["z"], {"x": slice(10, None)}, False, id="fit_indexer_x"
        ),
        pytest.param(0.1, None, ["z"], {"z": 0}, True, id="invalid_indexer_dim"),
    ],
)
def test_LimitedDataset(alpha, limit_only, feature_dims, fit_indexers, error):
    ds = get_dataset(alpha)
    expected_arr = get_expected_arr(alpha, feature_dims, fit_indexers)
    limit_only = limit_only or ds.data_vars
    expected_ds = ds.copy(deep=True)
    for var in limit_only:
        expected_ds[var] = xr.DataArray(expected_arr, dims=ds[var].dims)
    limiter = DatasetQuantileLimiter(alpha=alpha, limit_only=limit_only)
    if error:
        with pytest.raises(ValueError):
            limited_ds = limiter.fit_transform(
                ds, feature_dims=feature_dims, fit_indexers=fit_indexers
            )
    else:
        limited_ds = limiter.fit_transform(
            ds, feature_dims=feature_dims, fit_indexers=fit_indexers
        )
        for var in limited_ds:
            xr.testing.assert_allclose(limited_ds[var], expected_ds[var])
