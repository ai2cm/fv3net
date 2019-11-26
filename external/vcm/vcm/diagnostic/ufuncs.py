import numpy as np
import xarray as xr

"""
User defined functions for producing diagnostic outputs. For functions 
whose purpose is to calculate a new quantity, the output format must 
be a dataset with the new quantity stored as variable. 

Some of these replicate existing functions in vcm.calc, but wrapped so that
the result is the input dataset with diagnostic variable added.
"""


def compare_to_obs_time_series():
    pass


def remove_forecast_time_dim(ds):
    """
    The one step runs have a second time dimension 'forecast_time' that is used
    to calculate tendencies, but should be removed before plotting

    Args:
        ds: xarray dataset

    Returns:
        Same dataset but with extra time dim removed
    """
    if 'forecast_time' in ds.dims:
        ds = ds.isel(forecast_time=0).squeeze().drop('forecast_time')
    return ds


def pressure_thickness_weighted_mean(ds):
    pass


def mean_over_dim(ds, dim, var_to_avg, new_var):
    # note: apply always moves core dimensions to the end
    da_mean = xr.apply_ufunc(
        np.mean,
        ds[var_to_avg],
        input_core_dims=[[dim]],
        kwargs={'axis': -1},
        dask='allowed'
    )
    ds[new_var] = da_mean
    return ds


def sum_over_dim(ds, dim, var_to_sum, new_var):
    da_sum = xr.apply_ufunc(
        np.sum,
        ds[var_to_sum],
        input_core_dims=[[dim]],
        kwargs={'axis': -1},
        dask='allowed'
    )
    ds[new_var] = da_sum
    return ds
