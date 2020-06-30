import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from typing import Sequence
import xarray as xr

DIURNAL_VARS = ["Q1", "Q2"]
FLATTEN_DIMS = ["time", "x", "y", "tile",]


def create_diurnal_cycle_dataset(
    ds: xr.Dataset,
    longitude: xr.DataArray,
    diurnal_vars,
    n_bins: int = 24,
    time_dim: str = "time",
) -> xr.Dataset:
    """All the logic in this function deals with the case where the
    variable has an extra dimension that we want to keep in the final
    dataset for ease of merging.
    e.g. If there is are "target" and "predict" coords for integrated dQ1
    along a derivation dimension and we want the diurnal cycles of both
    computed separately.

    Args:
        ds (xr.Dataset)
        longitude (xr.DataArray): dataarray of lon values
        diurnal_vars ([type]): variables to compute diurnal cycle on
        n_bins (int, optional): Number bins for the 24 hr period. Defaults to 24.
        time_dim (str, optional): Name of time dim in dataset. Defaults to "time".

    Raises:
        ValueError: There can be at most one extra dimension seperate data arrays on
        before computing diurnal cycle.

    Returns:
        xr.Dataset: has dimension "local_time_hr" with bins as coordinates.
        Values are variable mean within each time bin.
    """
    ds_diurnal_cycle = xr.Dataset()
    for var in diurnal_vars:
        da = ds[var]
        additional_dim = [dim for dim in da.dims if dim not in FLATTEN_DIMS]
        if len(additional_dim) == 0:
            data_arrays = [da]
        elif len(additional_dim) == 1:
            additional_dim = additional_dim[0]
            coords = da[additional_dim].values
            data_arrays = [
                da.sel({additional_dim: coord}) for coord in coords]
        else:
            raise ValueError(
                "There should be at most one additional dimension to "
                "calculate diurnal cycle for variable along (e.g. derivation "
                "or data_source)."
            )
            
        var_diurnal_cycles = []
        for da in data_arrays:
            da_diurnal_cycle = bin_diurnal_cycle(
                da,
                longitude,
                n_bins,
                time_dim)
            var_diurnal_cycles.append(da_diurnal_cycle)

        if len(var_diurnal_cycles) > 1:
            ds_diurnal_cycle[var] = xr.concat(
                [da.expand_dims(additional_dim) for da in var_diurnal_cycles],
                dim=pd.Index(coords, name=additional_dim))
        else:
            ds_diurnal_cycle[var] = var_diurnal_cycles[0]
    return ds_diurnal_cycle


def bin_diurnal_cycle(
    da: xr.DataArray,
    longitude: xr.DataArray,
    n_bins: int = 24,
    time_dim: str = "time",
):
    bin_width_hrs = 24.0 / n_bins
    bin_centers = [i * bin_width_hrs / 2.0 for i in range(n_bins)]
    longitude = longitude.expand_dims({time_dim: da[time_dim].values})
    local_time = _local_time(longitude, time_dim)
    bin_means = _bin_diurnal_cycle(da, local_time, n_bins)
    da_diurnal_cycle = xr.DataArray(
        bin_means, dims=["local_time_hr"], coords={"local_time_hr": bin_centers}
    )
    return da_diurnal_cycle


def _local_time(da_lon: xr.DataArray, time_dim: str):
    hr_per_deg_lon = 1.0 / 15.0
    fractional_hr = (
        da_lon[time_dim].dt.hour
        + (da_lon[time_dim].dt.minute / 60.0)
        + (da_lon[time_dim].dt.second / 3600.0)
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
        statistic="mean",
        bins=bins,
    ).statistic
    bin_means = np.where(np.isnan(bin_means), 0.0, bin_means)
    return bin_means
