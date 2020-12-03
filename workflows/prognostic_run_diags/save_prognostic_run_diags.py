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
import argparse
import os
import sys

import datetime
import tempfile
import intake
import numpy as np
import xarray as xr
import shutil
from dask.diagnostics import ProgressBar

from toolz import curry
from collections import defaultdict
from typing import Dict, Callable, Mapping, Union

import vcm.catalog
import load_diagnostic_data as load_diags
import config
import diurnal_cycle
import transform
from constants import (
    HORIZONTAL_DIMS,
    DiagArg,
    GLOBAL_AVERAGE_DYCORE_VARS,
    GLOBAL_AVERAGE_PHYSICS_VARS,
    GLOBAL_BIAS_PHYSICS_VARS,
    DIURNAL_CYCLE_VARS,
    TIME_MEAN_VARS,
    RMSE_VARS,
)

import logging

logger = logging.getLogger("SaveDiags")

_DIAG_FNS = defaultdict(list)

DiagDict = Mapping[str, xr.DataArray]


def _prepare_diag_dict(
    suffix: str, new_diags: xr.Dataset, src_ds: xr.Dataset
) -> DiagDict:
    """
    Take a diagnostic dict add a suffix to all variable names and transfer attributes
    from a source dataset.  This is useful when the calculated diagnostics are a 1-to-1
    mapping from the source.
    """

    diags = {}
    for variable in new_diags:
        lower = variable.lower()
        da = new_diags[variable]
        if variable in src_ds:
            logger.debug(
                "Transferring available diagnostic attributes from source for "
                f"{variable}."
            )
            src_attrs = src_ds[variable].attrs
            da = da.assign_attrs(src_attrs)
        else:
            logger.debug(
                f"Not transferring attributes from source dataset for {variable}. This "
                "may cause issues with automated report generation."
            )

        diags[f"{lower}_{suffix}"] = da

    return diags


@curry
def diag_finalizer(var_suffix, func, *diag_args):
    """
    Wrapper to update dictionary to final variable names (with var_suffix)
    and attributes before returning.

    Expects diag_args to be of the form (prognostic_source, verif, grid)
    """

    def finalize(*diag_args):
        logger.debug(f"Finalizing wrapper to {func.__name__}")

        diags = func(*diag_args)
        prognostic, *_ = diag_args

        return _prepare_diag_dict(var_suffix, diags, prognostic)

    return finalize


@curry
def add_to_diags(
    diags_key: str, func: Callable[[DiagArg], DiagDict],
):
    """
    Add a function to the list of diagnostics to be computed
    for a specified group of data.

    Args:
        diags_key: A key for a group of diagnostics
        func: a function which computes a set of diagnostics.
            It needs to have the following signature::

                func(prognostic_run_data, verification_c48_data, grid)

            and should return diagnostics as a dict of xr.DataArrays.
            This output will be merged with all other decorated functions,
            so some care must be taken to avoid variable and coordinate clashes.
    """

    _DIAG_FNS[diags_key].append(func)

    return func


def compute_all_diagnostics(input_datasets: Dict[str, DiagArg]) -> DiagDict:
    """
    Compute all diagnostics for input data.

    Args:
        input_datasets: Input datasets with keys corresponding to the appropriate group
        of diagnostics (_DIAG_FNS) to be run on each data source.

    Returns:
        all computed diagnostics
    """

    diags = {}
    logger.info("Computing all diagnostics")

    for key, input_args in input_datasets.items():

        if key not in _DIAG_FNS:
            raise KeyError(f"No target diagnostics found for input data group: {key}")

        for func in _DIAG_FNS[key]:
            current_diags = func(*input_args)
            load_diags.warn_on_overwrite(diags.keys(), current_diags.keys())
            diags.update(current_diags)

    return diags


def rms(x, y, w, dims):
    return np.sqrt(((x - y) ** 2 * w).sum(dims) / w.sum(dims))


def bias(truth, prediction, w, dims):
    return ((prediction - truth) * w).sum(dims) / w.sum(dims)


def zonal_mean(
    ds: xr.Dataset, latitude: xr.DataArray, bins=np.arange(-90, 91, 2)
) -> xr.Dataset:
    zm = ds.groupby_bins(latitude, bins=bins).mean().rename(lat_bins="latitude")
    latitude_midpoints = [x.item().mid for x in zm["latitude"]]
    return zm.assign_coords(latitude=latitude_midpoints)


def time_mean(ds: xr.Dataset, dim: str = "time") -> xr.Dataset:
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


def dump_nc(ds: xr.Dataset, f):
    # to_netcdf closes file, which will delete the buffer
    # need to use a buffer since seek doesn't work with GCSFS file objects
    with tempfile.TemporaryDirectory() as dirname:
        url = os.path.join(dirname, "tmp.nc")
        ds.to_netcdf(url, engine="h5netcdf")
        with open(url, "rb") as tmp1:
            shutil.copyfileobj(tmp1, f)


@add_to_diags("dycore")
@diag_finalizer("rms_global")
@transform.apply("resample_time", "3H", inner_join=True)
@transform.apply("daily_mean", datetime.timedelta(days=10))
@transform.apply("subset_variables", RMSE_VARS)
def rms_errors(prognostic, verification_c48, grid):
    logger.info("Preparing rms errors")
    rms_errors = rms(prognostic, verification_c48, grid.area, dims=HORIZONTAL_DIMS)

    return rms_errors


@add_to_diags("dycore")
@diag_finalizer("zonal_and_time_mean")
@transform.apply("resample_time", "1H")
@transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
def zonal_means_dycore(prognostic, verification, grid):
    logger.info("Preparing zonal+time means (dycore)")
    zonal_means = zonal_mean(prognostic, grid.lat)
    return time_mean(zonal_means)


@add_to_diags("physics")
@diag_finalizer("zonal_and_time_mean")
@transform.apply("resample_time", "1H")
@transform.apply("subset_variables", GLOBAL_AVERAGE_PHYSICS_VARS)
def zonal_means_physics(prognostic, verification, grid):
    logger.info("Preparing zonal+time means (physics)")
    zonal_means = zonal_mean(prognostic, grid.lat)
    return time_mean(zonal_means)


@add_to_diags("dycore")
@diag_finalizer("zonal_bias")
@transform.apply("resample_time", "1H")
@transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
def zonal_mean_biases_dycore(prognostic, verification, grid):
    logger.info("Preparing zonal+time mean biases (dycore)")
    zonal_mean_bias = zonal_mean(prognostic - verification, grid.lat)
    return time_mean(zonal_mean_bias)


@add_to_diags("physics")
@diag_finalizer("zonal_bias")
@transform.apply("resample_time", "1H")
@transform.apply("subset_variables", GLOBAL_BIAS_PHYSICS_VARS)
def zonal_mean_biases_physics(prognostic, verification, grid):
    logger.info("Preparing zonal+time mean biases (physics)")
    zonal_mean_bias = zonal_mean(prognostic - verification, grid.lat)
    return time_mean(zonal_mean_bias)

###
for mask_type in ["global", "land", "sea", "tropics"]:
    @add_to_diags("dycore")
    @diag_finalizer(f"spatial_min_dycore_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H")
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
    def region_min_dycore(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing minimums for dycore variables ({mask_type})")
        return prognostic.min(dim=HORIZONTAL_DIMS)


for mask_type in ["global", "land", "sea", "tropics"]:
    @add_to_diags("dycore")
    @diag_finalizer(f"spatial_max_dycore_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H")
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
    def region_min_dycore(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing maximums for dycore variables ({mask_type})")
        return prognostic.max(dim=HORIZONTAL_DIMS)


for mask_type in ["global", "land", "sea", "tropics"]:
    @add_to_diags("physics")
    @diag_finalizer(f"spatial_min_physics_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H")
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_AVERAGE_PHYSICS_VARS)
    def region_min_dycore(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing minimums for physics variables ({mask_type})")
        return prognostic.min(dim=HORIZONTAL_DIMS)

        
for mask_type in ["global", "land", "sea", "tropics"]:
    @add_to_diags("physics")
    @diag_finalizer(f"spatial_max_physics_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H")
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_AVERAGE_PHYSICS_VARS)
    def region_min_dycore(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing maximums for physics variables ({mask_type})")
        return prognostic.max(dim=HORIZONTAL_DIMS)
###

for mask_type in ["global", "land", "sea", "tropics"]:

    @add_to_diags("dycore")
    @diag_finalizer(f"spatial_mean_dycore_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H")
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
    def global_averages_dycore(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing averages for dycore variables ({mask_type})")
        area_averages = (prognostic * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
            HORIZONTAL_DIMS
        )

        return area_averages


for mask_type in ["global", "land", "sea", "tropics"]:

    @add_to_diags("physics")
    @diag_finalizer(f"spatial_mean_physics_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H")
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_AVERAGE_PHYSICS_VARS)
    def global_averages_physics(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing averages for physics variables ({mask_type})")
        area_averages = (prognostic * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
            HORIZONTAL_DIMS
        )

        return area_averages


for mask_type in ["global", "land", "sea", "tropics"]:

    @add_to_diags("physics")
    @diag_finalizer(f"mean_bias_physics_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H", inner_join=True)
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_BIAS_PHYSICS_VARS)
    def global_biases_physics(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing average biases for physics variables ({mask_type})")
        bias_errors = bias(verification, prognostic, grid.area, HORIZONTAL_DIMS)

        return bias_errors


for mask_type in ["global", "land", "sea", "tropics"]:

    @add_to_diags("dycore")
    @diag_finalizer(f"mean_bias_dycore_{mask_type}")
    @transform.apply("mask_area", mask_type)
    @transform.apply("resample_time", "3H", inner_join=True)
    @transform.apply("daily_mean", datetime.timedelta(days=10))
    @transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
    def global_biases_dycore(prognostic, verification, grid, mask_type=mask_type):
        logger.info(f"Preparing average biases for dycore variables ({mask_type})")
        bias_errors = bias(verification, prognostic, grid.area, HORIZONTAL_DIMS)

        return bias_errors


@add_to_diags("physics")
@diag_finalizer("time_mean_value")
@transform.apply("resample_time", "1H", inner_join=True)
@transform.apply("subset_variables", TIME_MEAN_VARS)
def time_means_physics(prognostic, verification, grid):
    logger.info("Preparing time means for physics variables")
    return time_mean(prognostic)


@add_to_diags("physics")
@diag_finalizer("time_mean_bias")
@transform.apply("resample_time", "1H", inner_join=True)
@transform.apply("subset_variables", TIME_MEAN_VARS)
def time_mean_biases_physics(prognostic, verification, grid):
    logger.info("Preparing time mean biases for physics variables")
    return time_mean(prognostic - verification)


@add_to_diags("dycore")
@diag_finalizer("time_mean_value" c)
@transform.apply("resample_time", "1H", inner_join=True)
@transform.apply("subset_variables", TIME_MEAN_VARS)
def time_means_dycore(prognostic, verification, grid):
    logger.info("Preparing time means for physics variables")
    return time_mean(prognostic)


@add_to_diags("dycore")
@diag_finalizer("time_mean_bias")
@transform.apply("resample_time", "1H", inner_join=True)
@transform.apply("subset_variables", TIME_MEAN_VARS)
def time_mean_biases_dycore(prognostic, verification, grid):
    logger.info("Preparing time mean biases for physics variables")
    return time_mean(prognostic - verification)


for mask_type in ["global", "land", "sea"]:

    @add_to_diags("physics")
    @diag_finalizer(f"diurnal_{mask_type}")
    @transform.apply("mask_to_sfc_type", mask_type)
    @transform.apply("resample_time", "1H", time_slice=slice(24, -1), inner_join=True)
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


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--catalog", default=vcm.catalog.catalog_path)
    parser.add_argument(
        "--verification",
        help="Tag for simulation to use as verification data. Checks against "
        "'simulation' metadata from intake catalog.",
        default="40day_may2020",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = _get_parser().parse_args()

    attrs = vars(args)
    attrs["history"] = " ".join(sys.argv)

    catalog = intake.open_catalog(args.catalog)

    # get catalog entries for specified verification data
    verif_entries = config.get_verification_entries(args.verification, catalog)

    input_data = {
        "dycore": load_diags.load_dycore(args.url, verif_entries["dycore"], catalog),
        "physics": load_diags.load_physics(args.url, verif_entries["physics"], catalog),
    }

    # begin constructing diags
    diags = {}

    # maps
    diags["pwat_run_initial"] = input_data["dycore"][0].PWAT.isel(time=0)
    diags["pwat_run_final"] = input_data["dycore"][0].PWAT.isel(time=-2)
    diags["pwat_verification_final"] = input_data["dycore"][0].PWAT.isel(time=-2)

    diags.update(compute_all_diagnostics(input_data))

    # add grid vars
    diags = xr.Dataset(diags, attrs=attrs)
    diags = diags.merge(input_data["dycore"][2])

    logger.info("Forcing computation.")
    with ProgressBar():
        diags = diags.load()

    logger.info(f"Saving data to {args.output}")
    diags.to_netcdf(args.output)
