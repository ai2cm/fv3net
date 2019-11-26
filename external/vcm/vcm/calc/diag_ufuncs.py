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


def apply_pressure_thickness_weighting(
        ds,
        var_to_weight
):
    pressure_thickness_weights = ds.delp / ds.delp.sum('pfull')
    ds[var_to_weight] = ds[var_to_weight] * pressure_thickness_weights
    return ds


def mean_over_dim(
        ds,
        dim,
        var_to_avg,
        new_var,
        apply_delp_weighting=False
):
    if apply_delp_weighting:
        ds = apply_pressure_thickness_weighting(ds, var_to_avg)
    da_mean = ds[var_to_avg].mean(dim)
    ds[new_var] = da_mean
    return ds


def sum_over_dim(
        ds,
        dim,
        var_to_sum,
        new_var,
        apply_delp_weighting=False
):
    if apply_delp_weighting:
        ds = apply_pressure_thickness_weighting(ds, var_to_avg)
    da_sum = ds[var_to_sum].sum(dim)
    ds[new_var] = da_sum
    return ds
