""" User defined functions for producing diagnostic outputs.
For functions whose purpose is to calculate a new quantity,
the output format must be a dataset with the new quantity stored as variable.

Some of these replicate existing functions, but act as a wrapper so that
the function returns the input dataset with diagnostic variable added.
"""


def average_over_time_bin(ds, var, time_dim, sample_freq, new_var):
    """

    Args:
        ds: xarray dataset
        var: variable to take time mean of
        time_dim: time dimension name
        sample_freq: string (must be datetime-like), e.g. '30min', 2H', '1D'
        new_var: name for time averaged variable that will be added to dataset

    Returns:

    """
    da_var_time_mean = ds.resample(indexer={time_dim: sample_freq}).mean()[var]
    ds[new_var] = da_var_time_mean
    return ds


def remove_extra_dim(ds, extra_dim="forecast_time"):
    """ Sometimes dataarrays have extra dimensions that complicate plotting.
    e.g. The one step runs have a second time dimension 'forecast_time' that is used
    to calculate tendencies. However, carrying around the extra time dim after
    calculation complicates plotting, so it is removed before using the final
    dataarray in mapping functions

    Args:
        ds: xarray dataset

    Returns:
        Same dataset but with extra time dim removed
    """

    try:
        if len(ds[extra_dim].values) > 1:
            raise ValueError(
                f"Function remove_extra_dim should only be used on redundant dimensions \
                             of length 1. You tried to remove {extra_dim} \
                             which has length {len(ds[extra_dim].values)}."
            )
    except TypeError:
        # Appropriate usage of this function should raise a TypeError above
        pass

    if extra_dim in ds.dims:
        ds = ds.isel({extra_dim: 0}).squeeze().drop(extra_dim)
    return ds


def apply_weighting(ds, var_to_weight, weighting_var, weighting_dims):
    weights = ds[weighting_var] / ds[weighting_var].sum(weighting_dims)
    ds[var_to_weight] = ds[var_to_weight] * weights
    return ds


def mean_over_dim(
    ds, dim, var_to_avg, new_var,
):
    da_mean = ds[var_to_avg].mean(dim)
    ds[new_var] = da_mean
    return ds


def sum_over_dim(
    ds, dim, var_to_sum, new_var,
):
    da_sum = ds[var_to_sum].sum(dim)
    ds[new_var] = da_sum
    return ds


def mask_to_surface_type(ds, surface_type, surface_type_var="slmsk"):
    valid_surface_types = ["land", "sea", "seaice", "sea_ice", "sea ice"]
    if surface_type not in valid_surface_types:
        raise ValueError(
            f"Argument 'surface_type' must be one of {valid_surface_types}."
        )
    surface_type_codes = {"sea": 0, "land": 1, "seaice": 2, "sea_ice": 2, "sea ice": 2}
    mask = ds.isel(pfull=0)[surface_type_var == surface_type_codes[surface_type]]
    ds = ds[mask]
    return ds
