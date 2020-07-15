import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from typing import Sequence
import xarray as xr

from .config import VARNAMES
from .utils import snap_mask_to_type

FLATTEN_DIMS = [
    "time",
    "x",
    "y",
    "tile",
]
DIURNAL_CYCLE_DIM = "local_time_hr"
SURFACE_TYPE_DIM = "surface_type"


def create_diurnal_cycle_dataset(
    ds: xr.Dataset,
    longitude: xr.DataArray,
    diurnal_vars: Sequence[str],
    n_bins: int = 24,
    time_dim: str = "time",
    flatten_dims: Sequence[str] = FLATTEN_DIMS,
) -> xr.Dataset:
    """ Concats diurnal cycles for surface types and keeps additional dims
    if applicable.

    Args:
        ds: input dataset
        longitude: dataarray of lon values
        diurnal_vars: variables to compute diurnal cycle on
        n_bins: Number bins for the 24 hr period. Defaults to 24.
        time_dim: Name of time dim in dataset. Defaults to "time".

    Raises:
        ValueError: There can be at most one extra dimension along which to
        compute separate diurnal cycles for variable.
    Returns:
        Dataset with coords {"surface_type": surface type, "local_time_hr": time bins}.
        If there are "target"/"predict" coords for some variables, the output for those
        variables will also have this coordinate along the "derivation" dim.
        Data array values are the variable mean within each time bin.
    """
    var_sfc = VARNAMES["surface_type"]
    domain_datasets = {
        "global": _calc_diurnal_vars_with_extra_dims(
            ds, longitude, diurnal_vars, n_bins, time_dim, flatten_dims)
    }
    ds[var_sfc] = snap_mask_to_type(ds[var_sfc])
    for surface_type in ["land", "sea"]:
        domain_datasets[surface_type] = _calc_diurnal_vars_with_extra_dims(
            ds.where(ds[var_sfc] == surface_type),
            longitude,
            diurnal_vars,
            n_bins,
            time_dim,
            flatten_dims,
        )
    domains, datasets = [], []
    for key, value in domain_datasets.items():
        domains.append(key)
        datasets.append(value)
    return xr.concat(datasets, dim=pd.Index(domains, name=SURFACE_TYPE_DIM))


def _calc_diurnal_vars_with_extra_dims(
    ds: xr.Dataset,
    longitude: xr.DataArray,
    diurnal_vars: Sequence[str],
    n_bins: int = 24,
    time_dim: str = "time",
    flatten_dims: Sequence[str] = FLATTEN_DIMS,
) -> xr.Dataset:
    """All the logic in this function deals with the case where the
    variable has an extra dimension that we want to keep in the final
    dataset for ease of merging.
    e.g. If there are "target" and "predict" coords for integrated dQ1
    along a derivation dimension and we want the diurnal cycles of both
    computed separately.

    Args:
        ds: input dataset
        longitude: dataarray of lon values
        diurnal_vars: variables to compute diurnal cycle on
        n_bins: Number bins for the 24 hr period. Defaults to 24.
        time_dim: Name of time dim in dataset. Defaults to "time".

    Raises:
        ValueError: There can be at most one extra dimension along which to
        compute separate diurnal cycles for variable.
    Returns:
        Dataset with dimension "local_time_hr" and bins as coordinates.
        Data array values are the variable mean within each time bin.
    """
    ds_diurnal_cycle = xr.Dataset()
    for var in diurnal_vars:
        da = ds[var]
        additional_dim = [dim for dim in da.dims if dim not in flatten_dims]
        if len(additional_dim) == 0:
            data_arrays = [da]
        elif len(additional_dim) == 1:
            additional_dim = additional_dim[0]
            coords = da[additional_dim].values
            data_arrays = [da.sel({additional_dim: coord}) for coord in coords]
        else:
            raise ValueError(
                "There should be at most one additional dimension to "
                "calculate diurnal cycle for variable along (e.g. derivation "
                "or data_source)."
            )

        var_diurnal_cycles = []
        for da in data_arrays:
            da_diurnal_cycle = bin_diurnal_cycle(da, longitude, n_bins, time_dim)
            var_diurnal_cycles.append(da_diurnal_cycle)

        if len(var_diurnal_cycles) > 1:
            ds_diurnal_cycle[var] = xr.concat(
                [da.expand_dims(additional_dim) for da in var_diurnal_cycles],
                dim=pd.Index(coords, name=additional_dim),
            )
        else:
            ds_diurnal_cycle[var] = var_diurnal_cycles[0]
    return ds_diurnal_cycle


def bin_diurnal_cycle(
    da: xr.DataArray, longitude: xr.DataArray, n_bins: int = 24, time_dim: str = "time",
) -> xr.DataArray:
    """Bins the input variable data array into diurnal cycle with variable mean
    given in each time bin.

    Args:
        da: variable to calculate diurnal cycle of
        longitude: Longitude grid values
        n_bins: Number time bins in a day. Defaults to 24.
        time_dim: Time dim name in input data array. Defaults to "time".

    Returns:
        DataArray where coords are time bin starts, values are variable means in bins
    """
    bin_width_hrs = 24.0 / n_bins
    bin_centers = [i * bin_width_hrs for i in range(n_bins)]
    local_time = _local_time(longitude, da[time_dim])
    bin_means = _bin_diurnal_cycle(da, local_time, n_bins)
    da_diurnal_cycle = xr.DataArray(
        bin_means, dims=[DIURNAL_CYCLE_DIM], coords={DIURNAL_CYCLE_DIM: bin_centers}
    )
    return da_diurnal_cycle


def _local_time(da_lon: xr.DataArray, da_time: xr.DataArray) -> xr.DataArray:
    hr_per_deg_lon = 1.0 / 15.0
    fractional_hr = (
        da_time.dt.hour + (da_time.dt.minute / 60.0) + (da_time.dt.second / 3600.0)
    )
    local_time = (fractional_hr + da_lon * hr_per_deg_lon) % 24
    return local_time


def _bin_diurnal_cycle(
    da_var: xr.DataArray, local_time: xr.DataArray, n_bins,
):
    bins = np.linspace(0, 24, n_bins + 1)
    bin_means = binned_statistic(
        local_time.values.flatten(),
        da_var.values.flatten(),
        statistic=np.nanmean,
        bins=bins,
    ).statistic
    return bin_means
