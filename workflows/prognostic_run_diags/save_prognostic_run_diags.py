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
from functools import lru_cache
from typing import Tuple, Dict, Callable, Mapping, Sequence

import vcm
import load_diagnostic_data as load_diags
import diurnal

import logging

logger = logging.getLogger(__file__)

_DIAG_FNS = defaultdict(list)
_DERIVED_VAR_FNS = defaultdict(Callable)
_TRANSFORM_FNS = {}

HORIZONTAL_DIMS = ["x", "y", "tile"]
SECONDS_PER_DAY = 86400

DiagArg = Tuple[xr.Dataset, xr.Dataset, xr.Dataset]
DiagDict = Mapping[str, xr.DataArray]


def add_to_input_transform_fns(func):

    _TRANSFORM_FNS[func.__name__] = func

    return func


@add_to_input_transform_fns
@lru_cache(max_size=32)
def resample_time(arg: DiagArg, freq_label: str, time_slice=slice(None, -1)) -> DiagArg:
    prognostic, verification, grid = arg
    prognostic = prognostic.resample(time=freq_label, label="right").nearest()
    prognostic = prognostic.isel(time=time_slice)
    if "time" in verification:  # verification might be an empty dataset
        verification = verification.sel(time=prognostic.time)
    return prognostic, verification, grid


@add_to_input_transform_fns
@lru_cache(max_size=32)
def mask_to_sfc_type(
    arg: DiagArg, surface_type: str, mask_var_name: str = "SLMSKsfc"
) -> DiagArg:
    prognostic, verification, grid = arg
    masked_prognostic = vcm.mask_to_surface_type(
        prognostic, surface_type, surface_type_var=mask_var_name
    )
    masked_verification = vcm.mask_to_surface_type(
        verification, surface_type, surface_type=mask_var_name
    )

    return masked_prognostic, masked_verification, grid


def apply_transform(transform_params, func):

    transform_key, transform_args, transform_kwargs = transform_params
    if transform_key not in _TRANSFORM_FNS:
        raise KeyError(
            f"Unrecognized transform, {transform_key} requested for {func.__name__}"
        )

    transform_func = _TRANSFORM_FNS[transform_key]

    def transform(*args):

        transformed_args = transform_func(*args)

        return func(*transformed_args)

    return transform


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


def diag_finalizer(var_suffix, func):
    def finalize(prognostic, verification, grid):

        diags = func(prognostic, verification, grid)

        return _prepare_diag_dict(var_suffix, diags, prognostic)

    return finalize


@curry
def add_to_diags(
    diags_key: str,
    var_suffix: str,
    func: Callable[[DiagArg], DiagDict],
    input_transforms: Tuple[str, Sequence, Mapping] = None,
):
    """
    Add a function to the list of diagnostics to be computed
    for a specified group of data.

    Args:
        diags_key: A key for a group of diagnostics
        var_suffix:  A suffix passed to the diagnostic finalizer which appends
            to the end of all variable names in the diagnostic.  Useful for
            preventing overlap with diagnostics names that are 1-to-1 with the
            input data
        func: a function which computes a set of diagnostics.
            It needs to have the following signature::

                func(prognostic_run_data, verification_c48_data, grid)

            and should return diagnostics as a dict of xr.DataArrays.
            This output will be merged with all other decorated functions,
            so some care must be taken to avoid variable and coordinate clashes.
        input_pre_transforms: List of transform functions with arguments to
            apply to input data before diagnostic is calculated.  Each tuple
            should contain the following items: transform function name,
            transform arguments, transform keyword arguments.
    """
    _DIAG_FNS[diags_key].append(func)

    if input_transforms is not None:
        for transform_params in input_transforms:
            func = apply_transform(transform_params, func)

    # Prepare non-overlapping variable names and transfer attributes from source
    if var_suffix is not None:
        func = diag_finalizer(var_suffix, func)

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
            load_diags.warn_on_overwrite(diags, current_diags)
            diags.update(current_diags)

    return diags


@curry
def add_to_derived_vars(diags_key: str, func: Callable[[xr.Dataset], xr.Dataset]):
    """Add a function that computes additional derived variables from input datasets.
    These derived variables should be simple combinations of input variables, not
    reductions such as global means which are diagnostics to be computed later.

    Args:
        diags_key: a key for a group of inputs/diagnostics
        func: a function which adds new variables to a given dataset
    """
    _DERIVED_VAR_FNS[diags_key] = func


def compute_all_derived_vars(input_datasets: Dict[str, DiagArg]) -> Dict[str, DiagArg]:
    """Compute derived variables for all input data sources.

    Args:
        input_datasets: Input datasets with keys corresponding to the appropriate group
        of inputs/diagnostics.

    Returns:
        input datasets with derived variables added to prognostic and verification data
    """
    for key, func in _DERIVED_VAR_FNS.items():
        prognostic, verification, grid = input_datasets[key]
        logger.info(f"Preparing all derived variables for {key} prognostic data")
        prognostic = prognostic.merge(func(prognostic))
        logger.info(f"Preparing all derived variables for {key} verification data")
        verification = verification.merge(func(verification))
        input_datasets[key] = prognostic, verification, grid
    return input_datasets


@add_to_derived_vars("physics")
def derived_physics_variables(ds: xr.Dataset) -> xr.Dataset:
    """Compute derived variables for physics datasets"""
    arrays = []
    for func in [
        _column_pq1,
        _column_pq2,
        _column_dq1,
        _column_dq2,
        _column_q1,
        _column_q2,
    ]:
        try:
            arrays.append(func(ds))
        except (KeyError, AttributeError):  # account for ds[var] and ds.var notations
            logger.warning(f"Missing variable for calculation in {func.__name__}")
    return xr.merge(arrays)


def _column_pq1(ds: xr.Dataset) -> xr.DataArray:
    net_heating_arg_labels = [
        "DLWRFsfc",
        "DSWRFsfc",
        "ULWRFsfc",
        "ULWRFtoa",
        "USWRFsfc",
        "USWRFtoa",
        "DSWRFtoa",
        "SHTFLsfc",
        "PRATEsfc",
    ]
    net_heating_args = [ds[var] for var in net_heating_arg_labels]
    column_pq1 = vcm.net_heating(*net_heating_args)
    column_pq1.attrs = {
        "long_name": "<pQ1> column integrated heating from physics",
        "units": "W/m^2",
    }
    return column_pq1.rename("column_integrated_pQ1")


def _column_pq2(ds: xr.Dataset) -> xr.Dataset:
    evap = vcm.latent_heat_flux_to_evaporation(ds.LHTFLsfc)
    column_pq2 = SECONDS_PER_DAY * (evap - ds.PRATEsfc)
    column_pq2.attrs = {
        "long_name": "<pQ2> column integrated moistening from physics",
        "units": "mm/day",
    }
    return column_pq2.rename("column_integrated_pQ2")


def _column_dq1(ds: xr.Dataset) -> xr.Dataset:
    column_dq1 = ds.net_heating
    column_dq1.attrs = {
        "long_name": "<dQ1> column integrated heating from ML",
        "units": "W/m^2",
    }
    return column_dq1.rename("column_integrated_dQ1")


def _column_dq2(ds: xr.Dataset) -> xr.Dataset:
    column_dq2 = SECONDS_PER_DAY * ds.net_moistening
    column_dq2.attrs = {
        "long_name": "<dQ2> column integrated moistening from ML",
        "units": "mm/day",
    }
    return column_dq2.rename("column_integrated_dQ2")


def _column_q1(ds: xr.Dataset) -> xr.Dataset:
    column_q1 = _column_pq1(ds) + _column_dq1(ds)
    column_q1.attrs = {
        "long_name": "<Q1> column integrated heating from physics+ML",
        "units": "W/m^2",
    }
    return column_q1.rename("column_integrated_Q1")


def _column_q2(ds: xr.Dataset) -> xr.Dataset:
    column_q2 = _column_pq2(ds) + _column_dq2(ds)
    column_q2.attrs = {
        "long_name": "<Q2> column integrated moistening from physics+ML",
        "units": "mm/day",
    }
    return column_q2.rename("column_integrated_Q2")


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


transform_3h = ("resample_time", "3H", {})
transform_15min = ("resample_time", "15min", {})


@add_to_diags(
    diags_key="dycore", var_suffix="rms_global", input_transforms=[transform_3h]
)
def rms_errors(resampled, verification_c48, grid):
    logger.info("Preparing rms errors")
    rms_errors = rms(resampled, verification_c48, grid.area, dims=HORIZONTAL_DIMS)

    return rms_errors


@add_to_diags(
    diags_key="dycore", var_suffix="global_avg", input_transforms=[transform_3h]
)
def global_averages_dycore(resampled, verification, grid):
    logger.info("Preparing global averages for dycore variables")
    area_averages = (resampled * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
        HORIZONTAL_DIMS
    )

    return area_averages


@add_to_diags(
    diags_key="physics", var_suffix="global_phys_avg", input_transforms=[transform_3h]
)
def global_averages_physics(resampled, verification, grid):
    logger.info("Preparing global averages for physics variables")
    area_averages = (resampled * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
        HORIZONTAL_DIMS
    )

    return area_averages


# TODO: enable this diagnostic once SHiELD physics diags can be loaded efficiently
# @add_to_diags(diags_key="physics", var_suffix="bias_global_physics", input_transforms=[transform_3h])
def global_biases_physics(resampled, verification, grid):
    logger.info("Preparing global average biases for physics variables")
    bias_errors = bias(verification, resampled, grid.area, HORIZONTAL_DIMS)

    return bias_errors


for mask_type in ["global", "land", "sea"]:
    @add_to_diags(
        diags_key="physics",
        var_suffix=f"diurnal_{mask_type}",
        input_transforms=[transform_15min, ("mask_to_sfc_type", mask_type, {})]
    )
    def _diurnal_func(resampled, verification, grid):
        logger.info(
            f"Preparing diurnal cycle info for physics variables with mask={mask_type}"
        )
        diurnal_cycles = diurnal.diurnal_cycles(resampled, verification, grid)

        return diurnal_cycles


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
    loaded_data = {
        "dycore": load_diags.load_dycore(args.url, args.grid_spec, catalog),
        "physics": load_diags.load_physics(args.url, args.grid_spec, catalog),
    }

    # add derived variables
    input_data = compute_all_derived_vars(loaded_data)

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
