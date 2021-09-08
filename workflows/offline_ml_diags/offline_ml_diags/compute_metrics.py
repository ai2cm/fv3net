# TODO: combine offline diags and prognostic diags into same source tree
# for now, this will import registry and transforms from prognostic diags
from fv3net.diagnostics.prognostic_run.registry import Registry
import fv3net.diagnostics.prognostic_run.transform as transform
from fv3net.diagnostics.prognostic_run.constants import DiagArg

import logging
from typing import Sequence, Tuple, Dict

import xarray as xr

import numpy as np

logger = logging.getLogger(__name__)


HORIZONTAL_DIMS = ["x", "y", "tile"]


def _prepare_diag_dict(suffix: str, ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """
    Take a diagnostic dataset and add a suffix to all variable names and return as dict.
    """

    diags = {}
    for variable in ds:
        lower = str(variable).lower()
        da = ds[variable]
        diags[f"{lower}_{suffix}"] = da

    return diags


def merge_metrics(metrics: Sequence[Tuple[str, xr.Dataset]]):
    out: Dict[str, xr.DataArray] = {}
    for (name, ds) in metrics:
        out.update(_prepare_diag_dict(name, ds))
    # ignoring type error that complains if Dataset created from dict
    return xr.Dataset(out)  # type: ignore


metrics_registry = Registry(merge_metrics)


def compute_metrics(
    prediction: xr.Dataset,
    target: xr.Dataset,
    grid: xr.Dataset,
    delp: xr.DataArray,
    n_jobs: int = -1,
):
    return metrics_registry.compute(
        DiagArg(prediction, target, grid, delp), n_jobs=n_jobs
    )


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
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def mse_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing mean squared errors for 2D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        mse_area_weighted_avg = weighted_mean(
            (predicted - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return mse_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"mse_pressure_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def mse_3d(diag_arg, mask_type=mask_type):
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        logger.info(f"Preparing mean squared errors for 3D variables, {mask_type}")
        if len(predicted) == 0:
            return xr.Dataset()
        mse_area_weighted_avg = weighted_mean(
            (predicted - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return mse_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"variance_2d_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def variance_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing variance for 2D variables, {mask_type}")
        target, grid = diag_arg.verification, diag_arg.grid
        mean = weighted_mean(target, weights=grid.area, dims=HORIZONTAL_DIMS).mean(
            "time"
        )
        return weighted_mean(
            (mean - target) ** 2, weights=grid.area, dims=HORIZONTAL_DIMS
        ).mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"variance_pressure_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def variance_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing variance for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
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
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def bias_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing biases for 2D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        biases_area_weighted_avg = weighted_mean(
            predicted - target, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return biases_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"bias_pressure_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def bias_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing biases for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        biases_area_weighted_avg = weighted_mean(
            predicted - target, weights=grid.area, dims=HORIZONTAL_DIMS
        )
        return biases_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"bias_2d_zonal_avg_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def bias_zonal_avg_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing zonal avg biases for 2D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        zonal_avg_bias = zonal_mean(predicted - target, grid.lat)
        return zonal_avg_bias.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"bias_pressure_level_zonal_avg_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def bias_zonal_avg_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing zonal avg biases for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        zonal_avg_bias = zonal_mean(predicted - target, grid.lat)
        return zonal_avg_bias.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"mse_pressure_level_zonal_avg_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def mse_zonal_avg_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing zonal avg MSEs for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        zonal_avg_mse = zonal_mean((predicted - target) ** 2, grid.lat)
        return zonal_avg_mse.mean("time")


for mask_type in ["global", "sea", "land"]:

    @metrics_registry.register(f"variance_pressure_level_zonal_avg_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def variance_zonal_avg_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing zonal avg variance for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        mean = weighted_mean(target, weights=grid.area, dims=HORIZONTAL_DIMS).mean(
            "time"
        )
        zonal_avg_variance = zonal_mean((mean - target) ** 2, grid.lat)
        return zonal_avg_variance.mean("time")
