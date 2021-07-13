#!/usr/bin/env python
# coding: utf-8
"""
This script computes diagnostics for prognostic runs.

Diagnostics are multiple dimensional curves that can be visualized to give
a more detailed look into the data underlying the metrics.

Diagnostics are loaded and saved in groupings corresponding to the source of the data
and its intended use. E.g. the "dycore" grouping includes diagnostics output by the
dynamical core of the model and saved in `atmos_dt_atmos.tile*.nc` while the "physics"
grouping contains outputs from the physics routines (`sfc_dt_atmos.tile*.nc` and
`diags.zarr`).
"""
import sys

import datetime
import intake
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
import fsspec

from typing import Mapping, Union, Tuple, Sequence

import vcm

from fv3net.diagnostics.prognostic_run import load_run_data as load_diags
from fv3net.diagnostics.prognostic_run import diurnal_cycle
from fv3net.diagnostics.prognostic_run import transform
from fv3net.diagnostics.prognostic_run.constants import (
    HISTOGRAM_BINS,
    HORIZONTAL_DIMS,
    GLOBAL_AVERAGE_DYCORE_VARS,
    GLOBAL_AVERAGE_PHYSICS_VARS,
    GLOBAL_BIAS_PHYSICS_VARS,
    DIURNAL_CYCLE_VARS,
    TIME_MEAN_VARS,
    RMSE_VARS,
    PRESSURE_INTERPOLATED_VARS,
)
from .registry import Registry

import logging

logger = logging.getLogger("SaveDiags")


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


def merge_diags(diags: Sequence[Tuple[str, xr.Dataset]]) -> Mapping[str, xr.DataArray]:
    out = {}
    for name, ds in diags:
        out.update(_prepare_diag_dict(name, ds))
    return out


# all functions added to these registries must take three xarray Datasets as inputs
# (specifically, the prognostic run output, the verification data and the grid dataset)
# and return an xarray dataset containing one or more diagnostics.
registries = {
    "dycore": Registry(merge_diags),
    "physics": Registry(merge_diags),
    "3d": Registry(merge_diags),
}
# expressions not allowed in decorator calls, so need explicit variables for each here
registry_dycore = registries["dycore"]
registry_physics = registries["physics"]
registry_3d = registries["3d"]


def rms(x, y, w, dims):
    with xr.set_options(keep_attrs=True):
        return np.sqrt(((x - y) ** 2 * w).sum(dims) / w.sum(dims))


def bias(truth, prediction):
    with xr.set_options(keep_attrs=True):
        return prediction - truth


def weighted_mean(ds, weights, dims):
    with xr.set_options(keep_attrs=True):
        return (ds * weights).sum(dims) / weights.sum(dims)


def zonal_mean(
    ds: xr.Dataset, latitude: xr.DataArray, bins=np.arange(-90, 91, 2)
) -> xr.Dataset:
    with xr.set_options(keep_attrs=True):
        zm = ds.groupby_bins(latitude, bins=bins).mean().rename(lat_bins="latitude")
    latitude_midpoints = [x.item().mid for x in zm["latitude"]]
    return zm.assign_coords(latitude=latitude_midpoints)


def time_mean(ds: xr.Dataset, dim: str = "time") -> xr.Dataset:
    with xr.set_options(keep_attrs=True):
        result = ds.mean(dim)
    return _assign_diagnostic_time_attrs(result, ds)


def _get_time_attrs(ds: Union[xr.Dataset, xr.DataArray]) -> Mapping[str, str]:
    if "time" in ds.coords:
        start_time = str(ds.time.values[0])
        end_time = str(ds.time.values[-1])
        return {"diagnostic_start_time": start_time, "diagnostic_end_time": end_time}


def _assign_diagnostic_time_attrs(
    diagnostics_ds: xr.Dataset, source_ds: xr.Dataset
) -> xr.Dataset:
    for variable in diagnostics_ds:
        if variable in source_ds:
            attrs = _get_time_attrs(source_ds[variable])
            diagnostics_ds[variable] = diagnostics_ds[variable].assign_attrs(attrs)
    return diagnostics_ds


def _assign_source_attrs(
    diagnostics_ds: xr.Dataset, source_ds: xr.Dataset
) -> xr.Dataset:
    """Get attrs for each variable in diagnostics_ds from corresponding in source_ds."""
    for variable in diagnostics_ds:
        if variable in source_ds:
            attrs = source_ds[variable].attrs
            diagnostics_ds[variable] = diagnostics_ds[variable].assign_attrs(attrs)
    return diagnostics_ds


@registry_dycore.register("rms_global")
@transform.apply("resample_time", "3H", inner_join=True)
@transform.apply("daily_mean", datetime.timedelta(days=10))
@transform.apply("subset_variables", RMSE_VARS)
def rms_errors(prognostic, verification_c48, grid):
    logger.info("Preparing rms errors")
    rms_errors = rms(prognostic, verification_c48, grid.area, dims=HORIZONTAL_DIMS)

    return rms_errors


@registry_dycore.register("zonal_and_time_mean")
@transform.apply("resample_time", "1H")
@transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
def zonal_means_dycore(prognostic, verification, grid):
    logger.info("Preparing zonal+time means (dycore)")
    zonal_means = zonal_mean(prognostic, grid.lat)
    return time_mean(zonal_means)


@registry_physics.register("zonal_and_time_mean")
@transform.apply("resample_time", "1H")
@transform.apply("subset_variables", GLOBAL_AVERAGE_PHYSICS_VARS)
def zonal_means_physics(prognostic, verification, grid):
    logger.info("Preparing zonal+time means (physics)")
    zonal_means = zonal_mean(prognostic, grid.lat)
    return time_mean(zonal_means)


@registry_3d.register("pressure_level_zonal_time_mean")
@transform.apply("subset_variables", PRESSURE_INTERPOLATED_VARS)
@transform.apply("insert_absent_3d_output_placeholder")
@transform.apply("resample_time", "3H")
def zonal_means_3d(prognostic, verification, grid):
    logger.info("Preparing zonal+time means (3d)")
    with xr.set_options(keep_attrs=True):
        zonal_means = zonal_mean(prognostic, grid.lat)
        return time_mean(zonal_means)


@registry_3d.register("pressure_level_zonal_bias")
@transform.apply("subset_variables", PRESSURE_INTERPOLATED_VARS)
@transform.apply("insert_absent_3d_output_placeholder")
@transform.apply("resample_time", "3H")
def zonal_bias_3d(prognostic, verification, grid):
    logger.info("Preparing zonal mean bias (3d)")
    with xr.set_options(keep_attrs=True):
        zonal_mean_bias = zonal_mean(bias(verification, prognostic), grid.lat)
        return time_mean(zonal_mean_bias)


@registry_dycore.register("zonal_bias")
@transform.apply("resample_time", "1H")
@transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
def zonal_and_time_mean_biases_dycore(prognostic, verification, grid):
    logger.info("Preparing zonal+time mean biases (dycore)")

    zonal_mean_bias = zonal_mean(bias(verification, prognostic), grid.lat)
    return time_mean(zonal_mean_bias)


@registry_physics.register("zonal_bias")
@transform.apply("resample_time", "1H")
@transform.apply("subset_variables", GLOBAL_BIAS_PHYSICS_VARS)
def zonal_and_time_mean_biases_physics(prognostic, verification, grid):
    logger.info("Preparing zonal+time mean biases (physics)")
    zonal_mean_bias = zonal_mean(bias(verification, prognostic), grid.lat)
    return time_mean(zonal_mean_bias)


for variable_set in ["dycore", "physics"]:
    if variable_set == "dycore":
        subset_variables = GLOBAL_AVERAGE_DYCORE_VARS
        registry = registry_dycore
    else:
        subset_variables = GLOBAL_AVERAGE_PHYSICS_VARS
        registry = registry_physics

    @registry.register("zonal_mean_value")
    @transform.apply("resample_time", "3H", inner_join=True)
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", subset_variables)
    def zonal_mean_hovmoller(prognostic, verification, grid):
        logger.info(f"Preparing zonal mean values ({variable_set})")
        return zonal_mean(prognostic, grid.lat)

    @registry.register("zonal_mean_bias")
    @transform.apply("resample_time", "3H", inner_join=True)
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", subset_variables)
    def zonal_mean_bias_hovmoller(prognostic, verification, grid):
        logger.info(f"Preparing zonal mean biases ({variable_set})")
        return zonal_mean(bias(verification, prognostic), grid.lat)

    for mask_type in ["global", "land", "sea", "tropics"]:

        @registry.register(f"spatial_min_{variable_set}_{mask_type}")
        @transform.apply("mask_area", mask_type)
        @transform.apply("resample_time", "3H")
        @transform.apply("daily_mean", datetime.timedelta(days=10))
        @transform.apply("subset_variables", subset_variables)
        def spatial_min(prognostic, verification, grid, mask_type=mask_type):
            logger.info(f"Preparing minimum for variables ({mask_type})")
            masked = prognostic.where(~grid["area"].isnull())
            with xr.set_options(keep_attrs=True):
                return masked.min(dim=HORIZONTAL_DIMS)

        @registry.register(f"spatial_max_{variable_set}_{mask_type}")
        @transform.apply("mask_area", mask_type)
        @transform.apply("resample_time", "3H")
        @transform.apply("daily_mean", datetime.timedelta(days=10))
        @transform.apply("subset_variables", subset_variables)
        def spatial_max(prognostic, verification, grid, mask_type=mask_type):
            logger.info(f"Preparing maximum for variables ({mask_type})")
            masked = prognostic.where(~grid["area"].isnull())
            with xr.set_options(keep_attrs=True):
                return masked.max(dim=HORIZONTAL_DIMS)


for mask_type in ["global", "land", "sea", "tropics"]:

    @registry_dycore.register(f"spatial_mean_dycore_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H")
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
    def global_averages_dycore(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing averages for dycore variables ({mask_type})")
        return weighted_mean(prognostic, grid.area, HORIZONTAL_DIMS)


for mask_type in ["global", "land", "sea", "tropics"]:

    @registry_physics.register(f"spatial_mean_physics_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H")
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_AVERAGE_PHYSICS_VARS)
    def global_averages_physics(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing averages for physics variables ({mask_type})")
        return weighted_mean(prognostic, grid.area, HORIZONTAL_DIMS)


for mask_type in ["global", "land", "sea", "tropics"]:

    @registry_physics.register(f"mean_bias_physics_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H", inner_join=True)
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_BIAS_PHYSICS_VARS)
    def global_biases_physics(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing average biases for physics variables ({mask_type})")
        bias_errors = bias(verification, prognostic)
        mean_bias_errors = weighted_mean(bias_errors, grid.area, HORIZONTAL_DIMS)
        return mean_bias_errors


for mask_type in ["global", "land", "sea", "tropics"]:

    @registry_dycore.register(f"mean_bias_dycore_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H", inner_join=True)
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
    def global_biases_dycore(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing average biases for dycore variables ({mask_type})")
        bias_errors = bias(verification, prognostic)
        mean_bias_errors = weighted_mean(bias_errors, grid.area, HORIZONTAL_DIMS)
        return mean_bias_errors


@registry_physics.register("time_mean_value")
@transform.apply("resample_time", "1H", inner_join=True)
@transform.apply("subset_variables", TIME_MEAN_VARS)
def time_means_physics(prognostic, verification, grid):
    logger.info("Preparing time means for physics variables")
    return time_mean(prognostic)


@registry_physics.register("time_mean_bias")
@transform.apply("resample_time", "1H", inner_join=True)
@transform.apply("subset_variables", TIME_MEAN_VARS)
def time_mean_biases_physics(prognostic, verification, grid):
    logger.info("Preparing time mean biases for physics variables")
    return time_mean(bias(verification, prognostic))


@registry_dycore.register("time_mean_value")
@transform.apply("resample_time", "1H", inner_join=True)
@transform.apply("subset_variables", TIME_MEAN_VARS)
def time_means_dycore(prognostic, verification, grid):
    logger.info("Preparing time means for physics variables")
    return time_mean(prognostic)


@registry_dycore.register("time_mean_bias")
@transform.apply("resample_time", "1H", inner_join=True)
@transform.apply("subset_variables", TIME_MEAN_VARS)
def time_mean_biases_dycore(prognostic, verification, grid):
    logger.info("Preparing time mean biases for physics variables")
    return time_mean(bias(verification, prognostic))


for mask_type in ["global", "land", "sea"]:

    @registry_physics.register(f"diurnal_{mask_type}")
    @transform.apply("mask_to_sfc_type", mask_type)
    @transform.apply("resample_time", "1H", inner_join=True)
    @transform.apply("subset_variables", DIURNAL_CYCLE_VARS)
    def _diurnal_func(
        prognostic, verification, grid, mask_type=mask_type
    ) -> xr.Dataset:
        # mask_type is added as a kwarg solely to give the logging access to the info
        logger.info(
            f"Computing diurnal cycle info for physics variables with mask={mask_type}"
        )
        if len(prognostic.time) == 0:
            return xr.Dataset({})
        else:
            diag = diurnal_cycle.calc_diagnostics(prognostic, verification, grid).load()
            return _assign_diagnostic_time_attrs(diag, prognostic)


@registry_physics.register("histogram")
@transform.apply("resample_time", "3H", inner_join=True, method="mean")
@transform.apply("subset_variables", list(HISTOGRAM_BINS.keys()))
def compute_histogram(prognostic, verification, grid):
    logger.info("Computing histograms for physics diagnostics")
    counts = xr.Dataset()
    for varname in prognostic.data_vars:
        count, width = vcm.histogram(
            prognostic[varname], bins=HISTOGRAM_BINS[varname], density=True
        )
        counts[varname] = count
        counts[f"{varname}_bin_width"] = width
    return _assign_source_attrs(
        _assign_diagnostic_time_attrs(counts, prognostic), prognostic
    )


@registry_physics.register("hist_bias")
@transform.apply("resample_time", "3H", inner_join=True, method="mean")
@transform.apply("subset_variables", list(HISTOGRAM_BINS.keys()))
def compute_histogram_bias(prognostic, verification, grid):
    logger.info("Computing histogram biases for physics diagnostics")
    counts = xr.Dataset()
    for varname in prognostic.data_vars:
        prognostic_count, _ = vcm.histogram(
            prognostic[varname], bins=HISTOGRAM_BINS[varname], density=True
        )
        verification_count, _ = vcm.histogram(
            verification[varname], bins=HISTOGRAM_BINS[varname], density=True
        )
        counts[varname] = bias(verification_count, prognostic_count)
    return _assign_source_attrs(
        _assign_diagnostic_time_attrs(counts, prognostic), prognostic
    )


def register_parser(subparsers):
    parser = subparsers.add_parser("save", help="Compute the prognostic run diags.")
    parser.add_argument("url", help="Prognostic run output location.")
    parser.add_argument("output", help="Output path including filename.")
    parser.add_argument("--catalog", default=vcm.catalog.catalog_path)
    parser.add_argument(
        "--verification",
        help="Tag for simulation to use as verification data. Checks against "
        "'simulation' metadata from intake catalog.",
        default="40day_may2020",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Parallelism for the computation of diagnostics. "
        "Defaults to using all available cores. Can set to a lower fixed value "
        "if you are often running  into read errors when multiple processes "
        "access data concurrently.",
        default=-1,
    )
    parser.set_defaults(func=main)


def main(args):

    logging.basicConfig(level=logging.INFO)
    attrs = vars(args)
    attrs["history"] = " ".join(sys.argv)

    # begin constructing diags
    diags = {}
    catalog = intake.open_catalog(args.catalog)
    prognostic = load_diags.SegmentedRun(args.url, catalog)
    verification = load_diags.CatalogSimulation(args.verification, catalog)
    grid = load_diags.load_grid(catalog)
    input_data = load_diags.evaluation_pair_to_input_data(
        prognostic, verification, grid
    )

    # maps
    diags["pwat_run_initial"] = input_data["dycore"][0].PWAT.isel(time=0)
    diags["pwat_run_final"] = input_data["dycore"][0].PWAT.isel(time=-2)
    diags["pwat_verification_final"] = input_data["dycore"][0].PWAT.isel(time=-2)

    for key, (prog, verif, grid) in input_data.items():
        diags.update(registries[key].compute(prog, verif, grid, n_jobs=args.n_jobs))

    # add grid vars
    diags = xr.Dataset(diags, attrs=attrs)
    diags = diags.merge(input_data["dycore"][2])

    logger.info("Forcing remaining computation.")
    with ProgressBar():
        diags = diags.load()

    logger.info(f"Saving data to {args.output}")
    with fsspec.open(args.output, "wb") as f:
        vcm.dump_nc(diags, f)
