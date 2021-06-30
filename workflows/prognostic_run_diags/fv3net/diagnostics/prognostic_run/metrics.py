#!/usr/bin/env python
"""Compute and print metrics from the output of save_prognostic_run_diags.py

This functions computes a list of named scalar performance metrics, that are useful for
comparing across models.

Usage:

    metrics.py <diagnostics netCDF file>

"""
from typing import Callable, Mapping, Tuple
import numpy as np
import xarray as xr
from toolz import curry
from .constants import HORIZONTAL_DIMS, PERCENTILES
import json

_METRICS = []

GRID_VARS = ["lon", "lat", "lonb", "latb", "area"]
SURFACE_TYPE_CODES = {"sea": (0, 2), "land": (1,), "seaice": (2,)}

_GRAVITY = 9.8065
_EARTH_RADIUS = 6.378e6


def grab_diag(ds, name):
    replace_dict = {}
    for var in ds:
        match = "_" + name
        if var.endswith(match):
            replace_dict[var] = var[: -len(match)]

    if len(replace_dict) == 0:
        return xr.Dataset()

    return ds[list(replace_dict.keys())].rename(replace_dict)


def to_unit_quantity(val):
    return {"value": val.item(), "units": val.units}


def to_dict(ds: xr.Dataset):
    return {key: to_unit_quantity(ds[key]) for key in ds}


def prepend_to_key(d, prefix):
    return {prefix + key: val for key, val in d.items()}


def weighted_mean(ds, w, dims):
    return (ds * w).sum(dims) / w.sum(dims)


def _mask_array(
    region: str, arr: xr.DataArray, land_sea_mask: xr.DataArray,
) -> xr.DataArray:
    if region == "global":
        masked_arr = arr.copy()
    elif region in SURFACE_TYPE_CODES:
        masks = [land_sea_mask == code for code in SURFACE_TYPE_CODES[region]]
        mask_union = masks[0]
        for mask in masks[1:]:
            mask_union = np.logical_or(mask_union, mask)
        masked_arr = arr.where(mask_union)
    else:
        raise ValueError(f"Masking procedure for region '{region}' is not defined.")
    return masked_arr


@curry
def add_to_metrics(metricname: str, func: Callable[[xr.Dataset], xr.Dataset]):
    """Register a function to be used for computing metrics

    This function will be passed the diagnostics xarray dataset,
    and should return a Dataset of scalar quantities.

    See rmse_3day below for an example.

    """

    def myfunc(diags):
        metrics = func(diags)
        return prepend_to_key(to_dict(metrics), f"{metricname}/")

    _METRICS.append(myfunc)
    return func


def compute_all_metrics(diags: xr.Dataset) -> Mapping[str, float]:
    out = {}
    for metric in _METRICS:
        out.update(metric(diags))
    return out


for day in [3, 5]:

    @add_to_metrics(f"rmse_{day}day")
    def rmse_on_day(diags, day=day):
        rms_global = grab_diag(diags, "rms_global").drop(GRID_VARS, errors="ignore")
        if len(rms_global) == 0:
            return xr.Dataset()
        rms_global_daily = rms_global.resample(time="1D").mean()

        try:
            rms_on_day = rms_global_daily.isel(time=day)
        except IndexError:  # don't compute metric if run didn't make it to day
            rms_on_day = xr.Dataset()

        restore_units(rms_global, rms_on_day)
        return rms_on_day


@add_to_metrics("rmse_days_3to7_avg")
def rmse_days_3to7_avg(diags):
    rms_global = grab_diag(diags, "rms_global").drop(GRID_VARS, errors="ignore")
    if len(rms_global) == 0:
        return xr.Dataset()
    rms_global_daily = rms_global.resample(time="1D").mean()
    if rms_global_daily.sizes["time"] >= 7:
        rmse_days_3to7_avg = rms_global_daily.isel(time=slice(2, 7)).mean("time")
    else:  # don't compute metric if run didn't make it to 7 days
        rmse_days_3to7_avg = xr.Dataset()

    restore_units(rms_global, rmse_days_3to7_avg)
    return rmse_days_3to7_avg


@add_to_metrics("drift_3day")
def drift_3day(diags):
    averages = grab_diag(diags, "spatial_mean_dycore_global").drop(
        GRID_VARS, errors="ignore"
    )
    if len(averages) == 0:
        return xr.Dataset()

    daily = averages.resample(time="1D").mean()

    try:
        drift = (daily.isel(time=3) - daily.isel(time=0)) / 3
    except IndexError:  # don't compute metric if run didn't make it to 3 days
        drift = xr.Dataset()

    for variable in drift:
        orig_unit = averages[variable].attrs["units"]
        drift[variable].attrs["units"] = orig_unit + "/day"
    return drift


for mask_type in ["global", "land", "sea"]:

    @add_to_metrics(f"time_and_{mask_type}_mean_value")
    def time_and_global_mean_value(diags, mask_type=mask_type):
        time_mean_value = grab_diag(diags, "time_mean_value")
        if len(time_mean_value) == 0:
            return xr.Dataset()
        masked_area = _mask_array(mask_type, diags["area"], diags["land_sea_mask"])
        time_and_global_mean_value = weighted_mean(
            time_mean_value, masked_area, HORIZONTAL_DIMS
        )
        restore_units(time_mean_value, time_and_global_mean_value)
        return time_and_global_mean_value


for mask_type in ["global", "land", "sea"]:

    @add_to_metrics(f"time_and_{mask_type}_mean_bias")
    def time_and_domain_mean_bias(diags, mask_type=mask_type):
        time_mean_bias = grab_diag(diags, f"time_mean_bias")
        if len(time_mean_bias) == 0:
            return xr.Dataset()
        masked_area = _mask_array(mask_type, diags["area"], diags["land_sea_mask"])
        time_and_domain_mean_bias = weighted_mean(
            time_mean_bias, masked_area, HORIZONTAL_DIMS
        )
        restore_units(time_mean_bias, time_and_domain_mean_bias)
        return time_and_domain_mean_bias


for mask_type, suffix in zip(["global", "land", "sea"], ["", "_land", "_sea"]):
    # Omits 'global' suffix to avoid breaking change in map plots
    @add_to_metrics(f"rmse_of_time_mean{suffix}")
    def rmse_time_mean(diags, mask_type=mask_type):
        time_mean_bias = grab_diag(diags, f"time_mean_bias")
        if len(time_mean_bias) == 0:
            return xr.Dataset()
        masked_area = _mask_array(mask_type, diags["area"], diags["land_sea_mask"])
        rms_of_time_mean_bias = np.sqrt(
            weighted_mean(time_mean_bias ** 2, masked_area, HORIZONTAL_DIMS)
        )
        restore_units(time_mean_bias, rms_of_time_mean_bias)
        return rms_of_time_mean_bias


for percentile in PERCENTILES:

    @add_to_metrics(f"percentile_{percentile}")
    def percentile_metric(diags, percentile=percentile):
        histogram = grab_diag(diags, "histogram")
        if len(histogram) == 0:
            return xr.Dataset()
        percentiles = xr.Dataset()
        data_vars = [v for v in histogram.data_vars if not v.endswith("bin_width")]
        for varname in data_vars:
            percentiles[varname] = compute_percentile(
                percentile,
                histogram[varname].values,
                histogram[f"{varname}_bins"].values,
                histogram[f"{varname}_bin_width"].values,
            )
        restore_units(histogram, percentiles)
        return percentiles


@add_to_metrics("tropics_max_minus_min")
def itcz_mass_transport(diags):
    if "northward_wind_pressure_level_zonal_time_mean" not in diags:
        return xr.Dataset()
    northward_wind = diags["northward_wind_pressure_level_zonal_time_mean"]
    psi = mass_streamfunction(northward_wind).sel(pressure=slice(30000, 70000))
    psi_mid_troposphere = psi.weighted(psi.pressure).mean("pressure")
    lat_min, lat_max = itcz_edges(psi_mid_troposphere)
    mass_transport = psi_mid_troposphere.sel(
        latitude=lat_max
    ) - psi_mid_troposphere.sel(latitude=lat_min)
    return xr.Dataset({"psi_300_to_700": mass_transport.assign_attrs(units="Gkg/s")})


@add_to_metrics("tropical_ascent_region_mean")
def tropical_ascent_region_mean(diags):
    if "northward_wind_pressure_level_zonal_time_mean" not in diags:
        return xr.Dataset()
    zonal_mean_diags = grab_diag(diags, "zonal_and_time_mean")
    northward_wind = diags["northward_wind_pressure_level_zonal_time_mean"]
    psi = mass_streamfunction(northward_wind).sel(pressure=slice(30000, 70000))
    psi_mid_troposphere = psi.weighted(psi.pressure).mean("pressure")
    lat_min, lat_max = itcz_edges(psi_mid_troposphere)
    ascent_region_mean = zonal_mean_diags.sel(latitude=slice(lat_min, lat_max)).mean(
        "latitude"
    )
    restore_units(zonal_mean_diags, ascent_region_mean)
    return ascent_region_mean


def compute_percentile(
    percentile: float, freq: np.ndarray, bins: np.ndarray, bin_widths: np.ndarray
) -> float:
    """Compute percentile given normalized histogram.

    Args:
        percentile: value between 0 and 100
        freq: array of frequencies normalized by bin widths
        bins: values of left sides of bins
        bin_widths: values of bin widths

    Returns:
        value of distribution at percentile
    """
    cumulative_distribution = np.cumsum(freq * bin_widths)
    if np.abs(cumulative_distribution[-1] - 1) > 1e-6:
        raise ValueError(
            "The provided frequencies do not integrate to one. "
            "Ensure that histogram is computed with density=True."
        )
    bin_midpoints = bins + 0.5 * bin_widths
    closest_index = np.argmin(np.abs(cumulative_distribution - percentile / 100))
    return bin_midpoints[closest_index]


def mass_streamfunction(
    zonal_mean_northward_wind: xr.DataArray,
    lat: str = "latitude",
    plev: str = "pressure",
) -> xr.DataArray:
    pressure_thickness = zonal_mean_northward_wind[plev].diff(plev, label="lower")
    psi = (zonal_mean_northward_wind * pressure_thickness).cumsum(dim=plev)
    psi = 2 * np.pi * _EARTH_RADIUS * np.cos(np.deg2rad(psi[lat])) * psi / _GRAVITY
    return (psi / 1e9).assign_attrs(long_name="mass streamfunction", units="Gkg/s")


def itcz_edges(
    vertically_integrated_psi: xr.DataArray, lat: str = "latitude",
) -> Tuple[float, float]:
    lat_min = vertically_integrated_psi.sel({lat: slice(-30, 10)}).idxmin(lat).item()
    lat_max = vertically_integrated_psi.sel({lat: slice(-10, 30)}).idxmax(lat).item()
    return lat_min, lat_max


def restore_units(source, target):
    for variable in target:
        target[variable].attrs["units"] = source[variable].attrs["units"]


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "metrics",
        help="Compute metrics from verification diagnostics. "
        "Prints to standard output.",
    )
    parser.add_argument("input", help="netcdf file of prognostic_run_diags save.")
    parser.set_defaults(func=main)


def main(args):
    diags = xr.open_dataset(args.input)
    diags["time"] = diags.time - diags.time[0]
    metrics = compute_all_metrics(diags)
    # print to stdout, use pipes to save
    print(json.dumps(metrics, indent=4))
