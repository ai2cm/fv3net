import numpy as np
import xarray as xr
import pytest
from vcm.limit import DatasetQuantileLimiter

EPS = 1.0e-3


def get_dataset(scale):
    xscaling = np.arange(-scale, (scale + EPS))[:, np.newaxis]
    zscaling = (np.arange(scale, 0.0 - EPS, -1.0) / scale)[np.newaxis, :]
    data_scaled = xscaling * zscaling
    da = xr.DataArray(data_scaled, dims=["x", "z"])
    return xr.Dataset({"Q1": da, "Q2": da})


def get_expected_arr(arr, scale, feature_dims, fit_indexers):
    expected_arr = arr.copy()
    if feature_dims == ["z"]:
        if fit_indexers is None:
            upper = (scale - 1.0) * (np.arange(scale, 0.0 - EPS, -1.0) / scale)
            lower = -upper
            expected_arr[-1, :] = upper
            expected_arr[0, :] = lower
        else:
            upper = (scale - 0.5) * (np.arange(scale, 0.0 - EPS, -1.0) / scale)
            lower = 0.5 * (np.arange(scale, 0.0 - EPS, -1.0) / scale)
            expected_arr[-1, :] = upper
            expected_arr[0 : (int(scale) + 1), :] = lower
    elif feature_dims == ["x"]:
        upper = ((scale - 0.5) / scale) * np.arange(-scale, (scale + EPS))
        lower = (0.5 / scale) * np.arange(-scale, (scale + EPS))
        expected_arr[:, -1] = lower
        expected_arr[:, 0] = upper
    else:
        raise ValueError("Only 'x' and 'z' are tested as feature_dims.")
    return expected_arr


@pytest.mark.parametrize(
    ["alpha", "limit_only", "feature_dims", "fit_indexers"],
    [
        pytest.param(0.1, None, ["z"], None, id="alpha=0.1"),
        pytest.param(0.2, None, ["z"], None, id="alpha=0.2"),
        pytest.param(0.1, ["Q1"], ["z"], None, id="limit_only_Q1"),
        pytest.param(0.1, None, ["x"], None, id="feature_dim_x"),
        pytest.param(0.1, None, ["z"], {"x": slice(10, None)}, id="fit_x_subset"),
    ],
)
def test_LimitedDataset(alpha, limit_only, feature_dims, fit_indexers):
    scale = 1.0 / alpha
    ds = get_dataset(scale)
    arr = ds.Q1.values
    expected_ds = ds.copy()
    expected_arr = get_expected_arr(arr, scale, feature_dims, fit_indexers)
    for var in limit_only:
        expected_ds[var] = xr.DataArray(expected_arr, dims=ds[var].dims)
    limiter = DatasetQuantileLimiter(alpha=alpha)
    limited_ds = limiter.fit_transform(
        ds,
        limit_only=limit_only,
        feature_dims=feature_dims,
        fit_indexers=fit_indexers,
    )
    
    for var in limited_ds:
        xr.testing.assert_allclose(limited_ds[var], expected_ds[var])