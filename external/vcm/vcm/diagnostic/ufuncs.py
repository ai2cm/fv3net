import numpy as np
import xarray as xr


def test_func(ds):
    return ds


def compare_to_obs_time_series():
    pass


def mean(da, dim):
    # note: apply always moves core dimensions to the end
    da_mean = xr.apply_ufunc(
        np.mean,
        da,
        input_core_dims=[[dim]],
        kwargs={'axis': -1})
    return da_mean
