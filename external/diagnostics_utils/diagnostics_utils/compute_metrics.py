from diagnostics_utils.registry import Registry
import diagnostics_utils.transform as transform

import logging
from typing import Mapping, Sequence, Tuple

import xarray as xr

import numpy as np

logger = logging.getLogger(__name__)


HORIZONTAL_DIMS = ["x", "y", "tile"]


def merge_scalar_metrics(
    metrics: Sequence[Tuple[str, Mapping[str, float]]]
) -> Mapping[str, float]:
    merged = {}
    for metric in metrics:
        func = metric[0]
        for var, value in metric[1].items():
            merged[f"{func}_{var}"] = value
    return merged


def _prepare_diag_dict(suffix: str, ds: xr.Dataset) -> Mapping[str, xr.DataArray]:
    """
    Take a diagnostic dataset and add a suffix to all variable names and return as dict.
    """

    diags = {}
    for variable in ds:
        lower = variable.lower()
        da = ds[variable]
        diags[f"{lower}_{suffix}"] = da

    return diags


def merge_metrics(metrics: Sequence[Tuple[str, Mapping[str, xr.DataArray]]]):
    out = {}
    for name, ds in metrics:
        out.update(_prepare_diag_dict(name, ds))
    return xr.Dataset(out)


metrics_registry = Registry(merge_metrics)


def compute_metrics(
    prediction: xr.Dataset, target: xr.Dataset, grid: xr.Dataset, delp: xr.DataArray,
):
    return metrics_registry.compute(prediction, target, grid, delp)


def mse(x, y, w, dims):
    with xr.set_options(keep_attrs=True):
        return ((x - y) ** 2 * w).sum(dims) / w.sum(dims)


def bias(truth, prediction, w, dims):
    with xr.set_options(keep_attrs=True):
        return prediction - truth


def weighted_mean(ds, weights, dims):
    with xr.set_options(keep_attrs=True):
        return (ds * weights).sum(dims) / weights.sum(dims)


def zonal_mean(
    ds: xr.Dataset, latitude: xr.DataArray, bins=np.arange(-90, 91, 2)
) -> xr.Dataset:
    with xr.set_options(keep_attrs=True):
        zm = (
            ds.groupby_bins(latitude, bins=bins)
            .mean(skipna=True)
            .rename(lat_bins="latitude")
        )
    latitude_midpoints = [x.item().mid for x in zm["latitude"]]
    return zm.assign_coords(latitude=latitude_midpoints)


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"mse_2d_{mask_type}")
    @transform.apply("select_2d_variables")
    @transform.apply("mask_to_sfc_type", mask_type)
    def mse_2d(predicted, target, grid, mask_type=mask_type):
        logger.info(f"Preparing mean squared errors for 2D variables, {mask_type}")
        mse_area_weighted_avg = weighted_mean(
            (predicted - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return mse_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"mse_pressure_level_{mask_type}")
    @transform.apply("select_3d_variables")
    @transform.apply("regrid_zdim_to_pressure_levels")
    @transform.apply("mask_to_sfc_type", mask_type)
    def mse_3d(predicted, target, grid, mask_type=mask_type):
        logger.info(f"Preparing mean squared errors for 3D variables, {mask_type}")
        if len(predicted) == 0:
            return xr.Dataset()
        mse_area_weighted_avg = weighted_mean(
            (predicted - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return mse_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"variance_2d_{mask_type}")
    @transform.apply("select_2d_variables")
    @transform.apply("mask_to_sfc_type", mask_type)
    def variance_2d(predicted, target, grid, mask_type=mask_type):
        logger.info(f"Preparing variance for 2D variables, {mask_type}")
        mean = weighted_mean(target, weights=grid.area, dims=HORIZONTAL_DIMS).mean(
            "time"
        )
        return weighted_mean(
            (mean - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        ).mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"variance_pressure_level_{mask_type}")
    @transform.apply("select_3d_variables")
    @transform.apply("regrid_zdim_to_pressure_levels")
    @transform.apply("mask_to_sfc_type", mask_type)
    def variance_3d(predicted, target, grid, mask_type=mask_type):
        logger.info(f"Preparing variance for 3D variables, {mask_type}")
        if len(predicted) == 0:
            return xr.Dataset()
        mean = weighted_mean(target, weights=grid.area, dims=HORIZONTAL_DIMS).mean(
            "time"
        )
        return weighted_mean(
            (mean - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        ).mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"bias_2d_{mask_type}")
    @transform.apply("select_2d_variables")
    @transform.apply("mask_to_sfc_type", mask_type)
    def bias_2d(predicted, target, grid, mask_type=mask_type):
        logger.info(f"Preparing biases for 2D variables, {mask_type}")
        biases_area_weighted_avg = weighted_mean(
            predicted - target, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return biases_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"bias_pressure_level_{mask_type}")
    @transform.apply("select_3d_variables")
    @transform.apply("regrid_zdim_to_pressure_levels")
    @transform.apply("mask_to_sfc_type", mask_type)
    def bias_3d(predicted, target, grid, mask_type=mask_type):
        logger.info(f"Preparing biases for 3D variables, {mask_type}")
        if len(predicted) == 0:
            return xr.Dataset()
        biases_area_weighted_avg = weighted_mean(
            predicted - target, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return biases_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"bias_2d_zonal_avg_{mask_type}")
    @transform.apply("select_2d_variables")
    @transform.apply("mask_to_sfc_type", mask_type)
    def bias_zonal_avg_2d(predicted, target, grid, mask_type=mask_type):
        logger.info(f"Preparing zonal avg biases for 2D variables, {mask_type}")
        zonal_avg_bias = zonal_mean(predicted - target, grid.lat)
        return zonal_avg_bias.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"bias_pressure_level_zonal_avg_{mask_type}")
    @transform.apply("select_3d_variables")
    @transform.apply("regrid_zdim_to_pressure_levels")
    @transform.apply("mask_to_sfc_type", mask_type)
    def bias_zonal_avg_3d(predicted, target, grid, mask_type=mask_type):
        logger.info(f"Preparing zonal avg biases for 3D variables, {mask_type}")
        if len(predicted) == 0:
            return xr.Dataset()
        zonal_avg_bias = zonal_mean(predicted - target, grid.lat)
        return zonal_avg_bias.mean("time")
