from fv3net.diagnostics._shared.registry import Registry
import fv3net.diagnostics._shared.transform as transform
from fv3net.diagnostics._shared.constants import (
    DiagArg,
    COL_DRYING,
    WVP,
    HISTOGRAM_BINS,
)
import logging
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Dict
import xarray as xr

import vcm

logger = logging.getLogger(__name__)

DOMAINS = (
    "land",
    "sea",
    "global",
    "positive_net_precipitation",
    "negative_net_precipitation",
)
SURFACE_TYPE = "land_sea_mask"
SURFACE_TYPE_ENUMERATION = {0.0: "sea", 1.0: "land", 2.0: "sea"}
DERIVATION_DIM = "derivation"
COL_MOISTENING = "column_integrated_Q2"


def _prepare_diag_dict(suffix: str, ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """
    Take a diagnostic dataset and add a suffix to all variable names and return as dict.
    Useful in multiple merge functions passed to registries.
    """

    diags = {}
    for variable in ds:
        lower = str(variable).lower()
        da = ds[variable]
        diags[f"{lower}_{suffix}"] = da

    return diags


def merge_diagnostics(metrics: Sequence[Tuple[str, xr.Dataset]]):
    out: Dict[str, xr.DataArray] = {}
    for (name, ds) in metrics:
        out.update(_prepare_diag_dict(name, ds))
    # ignoring type error that complains if Dataset created from dict
    return xr.Dataset(out)  # type: ignore


diagnostics_registry = Registry(merge_diagnostics)


def compute_diagnostics(
    prediction: xr.Dataset,
    target: xr.Dataset,
    grid: xr.Dataset,
    delp: xr.DataArray,
    n_jobs: int = -1,
):
    return diagnostics_registry.compute(
        DiagArg(prediction, target, grid, delp), n_jobs=n_jobs,
    )


def _calc_ds_diurnal_cycle(ds):
    """
    Calculates the diurnal cycle for all variables.  Expects
    time dimension and longitude variable "lon".
    """
    local_time_ = vcm.local_time(ds, time="time", lon_var="lon")
    local_time_.attrs = {"long_name": "local time", "units": "hour"}
    ds["local_time"] = np.floor(local_time_)  # equivalent to hourly binning
    with xr.set_options(keep_attrs=True):
        diurnal_cycles = ds.drop("lon").groupby("local_time").mean()
    return diurnal_cycles


def mse(x, y, w, dims):
    with xr.set_options(keep_attrs=True):
        return ((x - y) ** 2 * w).sum(dims) / w.sum(dims)


def bias(truth, prediction, w, dims):
    with xr.set_options(keep_attrs=True):
        return prediction - truth


def weighted_mean(ds, weights, dims):
    with xr.set_options(keep_attrs=True):
        return (ds * weights).sum(dims) / weights.sum(dims)


def _compute_wvp_vs_q2_histogram(ds: xr.Dataset) -> xr.Dataset:
    counts = xr.Dataset()
    col_drying = -ds[COL_MOISTENING].rename(COL_DRYING)
    col_drying.attrs["units"] = ds[COL_MOISTENING].attrs.get("units")
    bins = [HISTOGRAM_BINS[WVP], HISTOGRAM_BINS[COL_DRYING]]

    counts, wvp_bins, q2_bins = vcm.histogram2d(ds[WVP], col_drying, bins=bins)
    return xr.Dataset(
        {
            f"{WVP}_versus_{COL_DRYING}": counts,
            f"{WVP}_bin_width": wvp_bins,
            f"{COL_DRYING}_bin_width": q2_bins,
        }
    )


def _assign_source_attrs(
    diagnostics_ds: xr.Dataset, source_ds: xr.Dataset
) -> xr.Dataset:
    """Get attrs for each variable in diagnostics_ds from corresponding in source_ds."""
    for variable in diagnostics_ds:
        if variable in source_ds:
            attrs = source_ds[variable].attrs
            diagnostics_ds[variable] = diagnostics_ds[variable].assign_attrs(attrs)
    return diagnostics_ds


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"mse_2d_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_area, mask_type)
    def mse_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing mean squared errors for 2D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        mse_area_weighted_avg = weighted_mean(
            (predicted - target) ** 2, weights=grid.area, dims=diag_arg.horizontal_dims
        )
        return mse_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"mse_pressure_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_area, mask_type)
    def mse_3d(diag_arg, mask_type=mask_type):
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        logger.info(
            "Preparing mean squared errors for 3D variables "
            f"on pressure levels, {mask_type}"
        )
        if len(predicted) == 0:
            return xr.Dataset()
        mse_area_weighted_avg = weighted_mean(
            (predicted - target) ** 2, weights=grid.area, dims=diag_arg.horizontal_dims
        )
        return mse_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"mse_model_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.mask_area, mask_type)
    def mse_3d_model_levels(diag_arg, mask_type=mask_type):
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        logger.info(
            "Preparing mean squared errors for 3D variables "
            f"on model levels, {mask_type}"
        )
        if len(predicted) == 0:
            return xr.Dataset()
        mse_area_weighted_avg = weighted_mean(
            (predicted - target) ** 2, weights=grid.area, dims=diag_arg.horizontal_dims
        )
        return mse_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"variance_2d_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_area, mask_type)
    def variance_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing variance for 2D variables, {mask_type}")
        target, grid = diag_arg.verification, diag_arg.grid
        mean = weighted_mean(
            target, weights=grid.area, dims=diag_arg.horizontal_dims
        ).mean("time")
        return weighted_mean(
            (mean - target) ** 2, weights=grid.area, dims=diag_arg.horizontal_dims
        ).mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"variance_pressure_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_area, mask_type)
    def variance_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing variance for 3D variables, on pressure {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        mean = weighted_mean(
            target, weights=grid.area, dims=diag_arg.horizontal_dims
        ).mean("time")
        return weighted_mean(
            (mean - target) ** 2, weights=grid.area, dims=diag_arg.horizontal_dims
        ).mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"variance_model_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.mask_area, mask_type)
    def variance_3d_model_levels(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing variance for 3D variables on model levels, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()
        mean = weighted_mean(
            target, weights=grid.area, dims=diag_arg.horizontal_dims
        ).mean("time")
        return weighted_mean(
            (mean - target) ** 2, weights=grid.area, dims=diag_arg.horizontal_dims
        ).mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"bias_2d_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_area, mask_type)
    def bias_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing biases for 2D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        biases_area_weighted_avg = weighted_mean(
            predicted - target, weights=grid.area, dims=diag_arg.horizontal_dims
        )
        return biases_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"bias_pressure_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_area, mask_type)
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
            predicted - target, weights=grid.area, dims=diag_arg.horizontal_dims
        )
        return biases_area_weighted_avg.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"bias_2d_zonal_avg_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    def bias_zonal_avg_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing zonal avg biases for 2D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        zonal_avg_bias = vcm.zonal_average_approximate(
            grid.lat, predicted - target, lat_name="latitude"
        )
        return zonal_avg_bias.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"bias_pressure_level_zonal_avg_{mask_type}")
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
        zonal_avg_bias = vcm.zonal_average_approximate(
            grid.lat, predicted - target, lat_name="latitude"
        )
        return zonal_avg_bias.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"mse_pressure_level_zonal_avg_{mask_type}")
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
        zonal_avg_mse = vcm.zonal_average_approximate(
            grid.lat, (predicted - target) ** 2, lat_name="latitude"
        )
        return zonal_avg_mse.mean("time")


for mask_type in ["global", "sea", "land"]:

    @diagnostics_registry.register(f"variance_pressure_level_zonal_avg_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    @transform.apply(transform.mask_area, mask_type)
    def variance_zonal_avg_3d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing zonal avg variance for 3D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) == 0:
            return xr.Dataset()

        zonal_mean = vcm.zonal_average_approximate(
            grid.lat, target, lat_name="latitude"
        ).mean("time")
        squares_zonal_mean = vcm.zonal_average_approximate(
            grid.lat, target ** 2, lat_name="latitude"
        ).mean("time")

        return squares_zonal_mean - zonal_mean ** 2


for mask_type in ["global", "land", "sea"]:

    @diagnostics_registry.register(f"diurnal_cycle_{mask_type}")
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    @transform.apply(transform.select_2d_variables)
    def diurnal_cycle(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing diurnal cycle over sfc type {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        ds = xr.concat(
            [predicted, target],
            dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
        )
        ds["lon"] = grid["lon"]
        return _calc_ds_diurnal_cycle(ds)


for mask_type in ["global", "land", "sea"]:

    @diagnostics_registry.register(f"time_domain_mean_2d_{mask_type}")
    @transform.apply(transform.select_2d_variables)
    @transform.apply(transform.mask_area, mask_type)
    def time_domain_mean_2d(diag_arg, mask_type=mask_type):
        logger.info(f"Preparing time means for 2D variables, {mask_type}")
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) > 0:
            ds = xr.concat(
                [predicted, target],
                dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
            )
            return weighted_mean(
                ds, weights=grid.area, dims=diag_arg.horizontal_dims
            ).mean("time")
        else:
            return xr.Dataset()


for mask_type in ["global", "land", "sea"]:

    @diagnostics_registry.register(f"time_domain_mean_pressure_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_area, mask_type)
    def time_domain_mean_pressure_level(diag_arg, mask_type=mask_type):
        logger.info(
            f"Preparing time means for 3D variables interpolated to pressure "
            f"levels, {mask_type}"
        )
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) > 0:
            ds = xr.concat(
                [predicted, target],
                dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
            )
            return weighted_mean(
                ds, weights=grid.area, dims=diag_arg.horizontal_dims
            ).mean("time")
        else:
            return xr.Dataset()


for mask_type in [
    "global",
    "land",
    "sea",
    "positive_net_precipitation",
    "negative_net_precipitation",
]:

    @diagnostics_registry.register(f"time_domain_mean_model_level_{mask_type}")
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.mask_area, mask_type)
    def time_domain_mean_model_level(diag_arg, mask_type=mask_type):
        logger.info(
            f"Preparing time means for 3D variables on model levels, {mask_type}"
        )
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) > 0:
            ds = xr.concat(
                [predicted, target],
                dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
            )
            return weighted_mean(
                ds, weights=grid.area, dims=diag_arg.horizontal_dims
            ).mean("time")
        else:
            return xr.Dataset()


for mask_type in ["global", "land", "sea"]:

    @diagnostics_registry.register(
        f"time_domain_mean_pressure_level_zonal_avg_{mask_type}"
    )
    @transform.apply(transform.select_3d_variables)
    @transform.apply(transform.regrid_zdim_to_pressure_levels)
    @transform.apply(transform.mask_area, mask_type)
    def time_domain_mean_pressure_level_zonal_avg(diag_arg, mask_type=mask_type):
        logger.info(
            f"Preparing time means for zonal mean 3D variables interpolated to "
            f"pressure levels, {mask_type}"
        )
        predicted, target, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(predicted) > 0:
            ds = xr.concat(
                [predicted, target],
                dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
            )
            return vcm.zonal_average_approximate(
                grid.lat, ds, lat_name="latitude"
            ).mean("time")
        else:
            return xr.Dataset()


@diagnostics_registry.register(f"time_mean_global")
def time_mean(diag_arg):
    logger.info(f"Preparing global time means")
    predicted, target = (
        diag_arg.prediction,
        diag_arg.verification,
    )
    if len(predicted) > 0:
        ds = xr.concat(
            [predicted, target],
            dim=pd.Index(["predict", "target"], name=DERIVATION_DIM),
        )
        return ds.mean("time")


@diagnostics_registry.register("hist_2d")
@transform.apply(transform.mask_to_sfc_type, "sea")
@transform.apply(transform.mask_to_sfc_type, "tropics20")
def compute_hist_2d(diag_arg: DiagArg):
    histograms = []
    if COL_MOISTENING in diag_arg.prediction and WVP in diag_arg.prediction:
        for ds in [diag_arg.prediction, diag_arg.verification]:
            logger.info("Computing joint histogram of water vapor path versus Q2")
            hist = _compute_wvp_vs_q2_histogram(ds).squeeze()
            histograms.append(_assign_source_attrs(hist, ds))
        return xr.concat(
            histograms, dim=pd.Index(["predict", "target"], name=DERIVATION_DIM)
        )
    else:
        return xr.Dataset()


@diagnostics_registry.register("histogram")
@transform.apply(transform.subset_variables, [WVP, COL_MOISTENING])
def compute_histogram(diag_arg: DiagArg):
    logger.info("Computing histograms for physics diagnostics")
    histograms = []
    if COL_MOISTENING in diag_arg.prediction and WVP in diag_arg.prediction:
        for ds in [diag_arg.prediction, diag_arg.verification]:
            ds[COL_DRYING] = -ds[COL_MOISTENING]
            ds[COL_DRYING].attrs["units"] = getattr(
                ds[COL_MOISTENING], "units", "mm/day"
            )
            counts = xr.Dataset()
            for varname in ds.data_vars:
                count, width = vcm.histogram(
                    ds[varname], bins=HISTOGRAM_BINS[varname.lower()], density=True
                )
                counts[varname] = count
                counts[f"{varname}_bin_width"] = width
            histograms.append(_assign_source_attrs(counts, ds))
        return xr.concat(
            histograms, dim=pd.Index(["predict", "target"], name=DERIVATION_DIM)
        )
    else:
        return xr.Dataset()
