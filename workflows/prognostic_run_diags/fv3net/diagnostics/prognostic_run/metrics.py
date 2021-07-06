#!/usr/bin/env python
"""Compute and print metrics from the output of save_prognostic_run_diags.py

This functions computes a list of named scalar performance metrics, that are useful for
comparing across models.

Usage:

    metrics.py <diagnostics netCDF file>

"""
from typing import Mapping, Sequence, Tuple
import numpy as np
import xarray as xr
from .constants import HORIZONTAL_DIMS, PERCENTILES
from .registry import Registry
import json

GRID_VARS = ["lon", "lat", "lonb", "latb", "area"]
SURFACE_TYPE_CODES = {"sea": (0, 2), "land": (1,), "seaice": (2,)}


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


def merge_metrics(metrics: Sequence[Tuple[str, xr.Dataset]]) -> Mapping[str, float]:
    out = {}
    for name, ds in metrics:
        out.update(prepend_to_key(to_dict(ds), f"{name}/"))
    return out


metrics_registry = Registry(merge_metrics)


for day in [3, 5]:

    @metrics_registry.register(f"rmse_{day}day")
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


@metrics_registry.register("rmse_days_3to7_avg")
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


@metrics_registry.register("drift_3day")
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

    @metrics_registry.register(f"time_and_{mask_type}_mean_value")
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

    @metrics_registry.register(f"time_and_{mask_type}_mean_bias")
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
    @metrics_registry.register(f"rmse_of_time_mean{suffix}")
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

    @metrics_registry.register(f"percentile_{percentile}")
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
            units = histogram[f"{varname}_bin_width"].attrs["units"]
            percentiles[varname].attrs.update({"units": units})
        return percentiles


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
    metrics = metrics_registry.compute(diags, n_jobs=1)
    # print to stdout, use pipes to save
    print(json.dumps(metrics, indent=4))
