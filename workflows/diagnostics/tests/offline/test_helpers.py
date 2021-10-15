import xarray as xr
import numpy as np

from fv3net.diagnostics.offline._helpers import (
    get_variable_indices,
    _count_features_2d,
)


def data_array_with_feature_dim(feature_dim: int, has_time_dim=True):
    da = xr.DataArray(
        data=np.arange(feature_dim * 8).reshape(1, 2, feature_dim, 2, 2),
        dims=["time", "tile", "z", "y", "x"],
    )
    if not has_time_dim:
        da = da.isel(time=0).squeeze(drop=True)
    if feature_dim == 1:
        da = da.isel(z=0).squeeze(drop=True)
    return da


def test_get_variable_indices():
    ds = xr.Dataset(
        {
            "dim_1": data_array_with_feature_dim(1, has_time_dim=True),
            "dim_10": data_array_with_feature_dim(10, has_time_dim=True),
            "dim_10_no_time": data_array_with_feature_dim(10, has_time_dim=False),
        }
    )
    variable_indices = get_variable_indices(ds, ["dim_1", "dim_10", "dim_10_no_time"])
    assert variable_indices == {
        "dim_1": (0, 1),
        "dim_10": (1, 11),
        "dim_10_no_time": (11, 21),
    }


def test_count_features_2d():
    SAMPLE_DIM_NAME = "axy"
    ds = xr.Dataset(
        data_vars={
            "a": xr.DataArray(np.zeros([10]), dims=[SAMPLE_DIM_NAME]),
            "b": xr.DataArray(np.zeros([10, 1]), dims=[SAMPLE_DIM_NAME, "b_dim"]),
            "c": xr.DataArray(np.zeros([10, 5]), dims=[SAMPLE_DIM_NAME, "c_dim"]),
        }
    )
    names = list(ds.data_vars.keys())
    assert len(names) == 3
    out = _count_features_2d(names, ds, sample_dim_name=SAMPLE_DIM_NAME)
    assert len(out) == len(names)
    for name in names:
        assert name in out
    assert out["a"] == 1
    assert out["b"] == 1
    assert out["c"] == 5
