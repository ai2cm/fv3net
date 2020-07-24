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

import tempfile
import intake
import numpy as np
import xarray as xr
import shutil
from dask.diagnostics import ProgressBar

from pathlib import Path
from toolz import curry
from collections import defaultdict
from typing import Dict, Callable, Mapping

import load_diagnostic_data as load_diags
import diurnal_cycle
import transform
from constants import (
    HORIZONTAL_DIMS,
    DiagArg,
    GLOBAL_AVERAGE_DYCORE_VARS,
    GLOBAL_AVERAGE_PHYSICS_VARS,
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
        attrs = da.attrs
        if not attrs and variable in src_ds:
            logger.debug(
                "Transferring missing diagnostic attributes from source for "
                f"{variable}."
            )
            src_attrs = src_ds[variable].attrs
            da = da.assign_attrs(src_attrs)
        else:
            logger.debug(
                f"Diagnostic variable ({variable}) missing attributes. This "
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
@transform.apply("resample_time", "3H")
def rms_errors(resampled, verification_c48, grid):
    logger.info("Preparing rms errors")
    rms_errors = rms(resampled, verification_c48, grid.area, dims=HORIZONTAL_DIMS)

    return rms_errors


@add_to_diags("dycore")
@diag_finalizer("global_avg")
@transform.apply("resample_time", "3H")
@transform.apply("subset_variables", GLOBAL_AVERAGE_DYCORE_VARS)
def global_averages_dycore(resampled, verification, grid):
    logger.info("Preparing global averages for dycore variables")
    area_averages = (resampled * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
        HORIZONTAL_DIMS
    )

    return area_averages


@add_to_diags("physics")
@diag_finalizer("global_phys_avg")
@transform.apply("resample_time", "3H")
@transform.apply("subset_variables", GLOBAL_AVERAGE_PHYSICS_VARS)
def global_averages_physics(resampled, verification, grid):
    logger.info("Preparing global averages for physics variables")
    area_averages = (resampled * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
        HORIZONTAL_DIMS
    )

    return area_averages


@add_to_diags("physics")
@diag_finalizer("bias_global_physics")
@transform.apply("resample_time", "3H")
@transform.apply("subset_variables", GLOBAL_AVERAGE_PHYSICS_VARS)
def global_biases_physics(resampled, verification, grid):
    logger.info("Preparing global average biases for physics variables")
    bias_errors = bias(verification, resampled, grid.area, HORIZONTAL_DIMS)

    return bias_errors


for mask_type in ["global", "land", "sea"]:

    @add_to_diags("physics")
    @diag_finalizer(f"diurnal_{mask_type}")
    @transform.apply("mask_to_sfc_type", mask_type)
    @transform.apply("resample_time", "15min", time_slice=slice(96, -1))
    def _diurnal_func(resampled, verification, grid, mask_type=mask_type):
        # mask_type is added as a kwarg solely to give the logging access to the info
        logger.info(
            f"Preparing diurnal cycle info for physics variables with mask={mask_type}"
        )
        diurnal = diurnal_cycle.calc_diagnostics(resampled, verification, grid)

        return diurnal


def _catalog():
    TOP_LEVEL_DIR = Path(os.path.abspath(__file__)).parent.parent.parent
    return str(TOP_LEVEL_DIR / "catalog.yml")


if __name__ == "__main__":

    CATALOG = _catalog()

    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument(
        "--grid-spec", default="./grid_spec",
    )
    parser.add_argument("--catalog", default=CATALOG)

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    attrs = vars(args)
    attrs["history"] = " ".join(sys.argv)

    catalog = intake.open_catalog(args.catalog)
    input_data = {
        "dycore": load_diags.load_dycore(args.url, args.grid_spec, catalog),
        "physics": load_diags.load_physics(args.url, args.grid_spec, catalog),
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
