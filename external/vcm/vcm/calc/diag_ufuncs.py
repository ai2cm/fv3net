import numpy as np
import xarray as xr

"""
User defined functions for producing diagnostic outputs. For functions 
whose purpose is to calculate a new quantity, the output format must 
be a dataset with the new quantity stored as variable. 

Some of these replicate existing functions in vcm.calc, but wrapped so that
the result is the input dataset with diagnostic variable added.
"""


def remove_extra_dim(ds, extra_dim='forecast_time'):
    """ Sometimes dataarrays have extra dimensions that complicate plotting.
    e.g. The one step runs have a second time dimension 'forecast_time' that is used
    to calculate tendencies. However, carrying around the extra time dim after calculation
    complicates plotting, so it is removed before using the final dataarray in mapping functions

    Args:
        ds: xarray dataset

    Returns:
        Same dataset but with extra time dim removed
    """
    if len(ds[extra_dim].values) > 1:
        raise ValueError("Function remove_extra_dim should only be used on redundant dimensions \
                         of length 1. You tried to remove {0} which has length {1}."
                         .format(extra_dim, len(ds[extra_dim].values)))

    if extra_dim in ds.dims:
        ds = ds.isel({extra_dim: 0}).squeeze().drop(extra_dim)
    return ds


def apply_weighting(
        ds,
        var_to_weight,
        weighting_var,
        weighting_dims
):
    weights = (ds[weighting_var] / ds[weighting_var].sum(weighting_dims))
    ds[var_to_weight] = ds[var_to_weight] * weights
    return ds


def mean_over_dim(
        ds,
        dim,
        var_to_avg,
        new_var,
):
    da_mean = ds[var_to_avg].mean(dim)
    ds[new_var] = da_mean
    return ds


def sum_over_dim(
        ds,
        dim,
        var_to_sum,
        new_var,
):
    da_sum = ds[var_to_sum].sum(dim)
    ds[new_var] = da_sum
    return ds
