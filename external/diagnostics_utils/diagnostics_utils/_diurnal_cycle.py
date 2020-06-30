import numpy as np
from scipy.stats import binned_statistic
from typing import Sequence
import xarray as xr


def aggregate_diurnal_cycle(
    ds: xr.Dataset,
    longitude: xr.DataArray,
    diurnal_vars: Sequence[str],
    n_bins: int = 24,
    time_dim: str = "time",
):
    bin_width_hrs = 24.0 / n_bins
    bin_centers = [i * bin_width_hrs / 2.0 for i in range(n_bins)]
    ds_diurnal_cycle = xr.Dataset()
    for var in diurnal_vars:
        da = ds[var]
        longitude = longitude.expand_dims({time_dim: da[time_dim].values})
        local_time = _local_time(longitude, time_dim)
        bin_means = _bin_diurnal_cycle(da, local_time, n_bins)
        ds_diurnal_cycle[var] = xr.DataArray(
            bin_means, dims=["local_time_hr"], coords={"local_time_hr": bin_centers}
        )
    return ds_diurnal_cycle


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
