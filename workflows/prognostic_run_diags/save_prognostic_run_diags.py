#!/usr/bin/env python
# coding: utf-8
"""
This script computes diagnostics for prognostic runs.

Diagnostics are multiple dimensional curves that can be visualized to give
a more detailed look into the data underlying the metrics.
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
from typing import Tuple, Dict, Callable, Mapping

import vcm
import load_diagnostic_data as load_diags

import logging

logger = logging.getLogger(__file__)

_DIAG_FNS = defaultdict(list)
_DERIVED_VAR_FNS = defaultdict(Callable)

HORIZONTAL_DIMS = ["x", "y", "tile"]
SECONDS_PER_DAY = 86400

DiagArg = Tuple[xr.Dataset, xr.Dataset, xr.Dataset]
DiagDict = Mapping[str, xr.DataArray]


@curry
def add_to_diags(diags_key: str, func: Callable[[DiagArg], DiagDict]):
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
        of inputs/diagnostics..

    Returns:
        input datasets with derived variables added to prognostic and verification data
    """
    for key, func in _DERIVED_VAR_FNS.items():
        prognostic, verification, grid = input_datasets[key]
        logger.info(f"Preparing all derived variables for {key} prognostic data")
        prognostic = func(prognostic)
        logger.info(f"Preparing all derived variables for {key} verification data")
        verification = func(verification)
        input_datasets[key] = prognostic, verification, grid
    return input_datasets


@add_to_derived_vars("physics")
def derived_physics_variables(ds: xr.Dataset) -> xr.Dataset:
    """Compute derived variables for physics datasets"""
    for func in [
        _column_pq1,
        _column_pq2,
        _column_dq1,
        _column_dq2,
        _column_q1,
        _column_q2,
    ]:
        ds = func(ds)
    return ds


def _column_pq1(ds: xr.Dataset) -> xr.Dataset:
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
    if not _ds_contains(ds, set(net_heating_arg_labels)):
        return ds
    net_heating_args = [ds[var] for var in net_heating_arg_labels]
    column_pq1 = vcm.net_heating(*net_heating_args)
    column_pq1.attrs = {
        "long_name": "<pQ1> column integrated heating from physics",
        "units": "W/m^2",
    }
    return ds.update({"column_integrated_pQ1": column_pq1})


def _column_pq2(ds: xr.Dataset) -> xr.Dataset:
    if not _ds_contains(ds, {"LHTFLsfc", "PRATEsfc"}):
        return ds
    evap = vcm.latent_heat_flux_to_evaporation(ds.LHTFLsfc)
    column_pq2 = SECONDS_PER_DAY * (evap - ds.PRATEsfc)
    column_pq2.attrs = {
        "long_name": "<pQ2> column integrated moistening from physics",
        "units": "mm/day",
    }
    return ds.update({"column_integrated_pQ2": column_pq2})


def _column_dq1(ds: xr.Dataset) -> xr.Dataset:
    if not _ds_contains(ds, {"net_heating"}):
        return ds
    column_dq1 = ds.net_heating
    column_dq1.attrs = {
        "long_name": "<dQ1> column integrated heating from ML",
        "units": "W/m^2",
    }
    return ds.drop_vars("net_heating").update({"column_integrated_dQ1": column_dq1})


def _column_dq2(ds: xr.Dataset) -> xr.Dataset:
    if not _ds_contains(ds, {"net_moistening"}):
        return ds
    column_dq2 = SECONDS_PER_DAY * ds.net_moistening
    column_dq2.attrs = {
        "long_name": "<dQ2> column integrated moistening from ML",
        "units": "mm/day",
    }
    return ds.drop_vars("net_moistening").update({"column_integrated_dQ2": column_dq2})


def _column_q1(ds: xr.Dataset) -> xr.Dataset:
    if not _ds_contains(ds, {"column_integrated_dQ1", "column_integrated_pQ1"}):
        return ds
    column_q1 = ds.column_integrated_dQ1 + ds.column_integrated_pQ1
    column_q1.attrs = {
        "long_name": "<Q1> column integrated heating from physics+ML",
        "units": "W/m^2",
    }
    return ds.update({"column_integrated_Q1": column_q1})


def _column_q2(ds: xr.Dataset) -> xr.Dataset:
    if not _ds_contains(ds, {"column_integrated_dQ2", "column_integrated_pQ2"}):
        return ds
    column_q2 = ds.column_integrated_dQ2 + ds.column_integrated_pQ2
    column_q2.attrs = {
        "long_name": "<Q2> column integrated moistening from physics+ML",
        "units": "mm/day",
    }
    return ds.update({"column_integrated_Q2": column_q2})


def _ds_contains(ds: xr.Dataset, variables: set) -> bool:
    data_vars = set(ds.data_vars)
    ds_contains_variables = variables.issubset(data_vars)
    if not ds_contains_variables:
        logger.warning(f"A dataset is missing variables {variables - data_vars}")
    return ds_contains_variables


def rms(x, y, w, dims):
    return np.sqrt(((x - y) ** 2 * w).sum(dims) / w.sum(dims))


def bias(truth, prediction, w, dims):
    return ((prediction - truth) * w).sum(dims) / w.sum(dims)


def calc_ds_diurnal_cycle(ds):
    """
    Calculates the diurnal cycle on moisture variables.  Expects
    time dimension and longitude variable "grid_lont".
    """
    # TODO: switch to vcm.safe dataset usage
    moist_vars = ["LHTFLsfc", "PRATEsfc", "net_moistening"]
    local_time = vcm.local_time(ds, time="time", lon_var="grid_lont")

    ds = ds[[var for var in moist_vars if var in ds]]
    local_time = np.floor(local_time)  # equivalent to hourly binning
    ds["mean_local_time"] = local_time
    # TODO: groupby is pretty slow, appears to be single-threaded op
    diurnal_ds = ds.groupby("mean_local_time").mean()

    return diurnal_ds


def _add_derived_moisture_diurnal_quantities(ds_run, ds_verif):
    """
    Adds moisture quantites for the individual component comparison
    of the diurnal cycle  moisture
    """

    # For easy filtering into plot component
    filter_flag = "moistvar_comp"

    ds_verif[f"total_P_{filter_flag}"] = ds_verif["PRATEsfc"].assign_attrs(
        {"long_name": "diurnal precipitation from verification", "units": "kg/m^2/s"}
    )

    total_P = ds_run["PRATEsfc"]
    if "net_moistening" in ds_run:
        total_P = total_P - ds_run["net_moistening"]  # P - dQ2
    ds_run[f"total_P_{filter_flag}"] = total_P.assign_attrs(
        {
            "long_name": "diurnal precipitation from coarse model run",
            "units": "kg/m^2/s",
        }
    )

    E_run = vcm.latent_heat_flux_to_evaporation(ds_run["LHTFLsfc"])
    E_verif = vcm.latent_heat_flux_to_evaporation(ds_verif["LHTFLsfc"])
    E_diff = E_run - E_verif
    P_diff = ds_run[f"total_P_{filter_flag}"] - ds_verif[f"total_P_{filter_flag}"]

    ds_run[f"evap_diff_from_verif_{filter_flag}"] = E_diff.assign_attrs(
        {"long_name": "diurnal diff: evap_coarse - evap_hires", "units": "kg/m^2/s"}
    )
    ds_run[f"precip_diff_from_verif_{filter_flag}"] = P_diff.assign_attrs(
        {"long_name": "diurnal diff: precip_coarse - precip_hires", "units": "kg/m^2/s"}
    )

    net_precip_diff = P_diff - E_diff
    ds_run[f"net_precip_diff_from_verif_{filter_flag}"] = net_precip_diff.assign_attrs(
        {
            "long_name": "diurnal diff: net_precip_coarse - net_precip_hires",
            "units": "kg/m^2/s",
        }
    )

    return ds_run, ds_verif


def dump_nc(ds: xr.Dataset, f):
    # to_netcdf closes file, which will delete the buffer
    # need to use a buffer since seek doesn't work with GCSFS file objects
    with tempfile.TemporaryDirectory() as dirname:
        url = os.path.join(dirname, "tmp.nc")
        ds.to_netcdf(url, engine="h5netcdf")
        with open(url, "rb") as tmp1:
            shutil.copyfileobj(tmp1, f)


def _vars_to_diags(suffix: str, ds: xr.Dataset, src_ds: xr.Dataset) -> DiagDict:
    # converts dataset variables into diagnostics dict
    # and assigns attrs from src_ds if none are set

    diags = {}
    for variable in ds:
        lower = variable.lower()
        da = ds[variable]
        attrs = da.attrs
        if not attrs:
            src_attrs = src_ds[variable].attrs
            da = da.assign_attrs(src_attrs)

        diags[f"{lower}_{suffix}"] = da

    return diags


@add_to_diags("dycore")
def rms_errors(resampled, verification_c48, grid):
    logger.info("Preparing rms errors")
    rms_errors = rms(resampled, verification_c48, grid.area, dims=HORIZONTAL_DIMS)

    return _vars_to_diags("rms_global", rms_errors, resampled)


@add_to_diags("dycore")
def global_averages_dycore(resampled, verification, grid):
    logger.info("Preparing global averages for dycore variables")
    area_averages = (resampled * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
        HORIZONTAL_DIMS
    )

    return _vars_to_diags("global_avg", area_averages, resampled)


@add_to_diags("physics")
def global_averages_physics(resampled, verification, grid):
    logger.info("Preparing global averages for physics variables")
    area_averages = (resampled * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
        HORIZONTAL_DIMS
    )

    return _vars_to_diags("global_phys_avg", area_averages, resampled)


# TODO: enable this diagnostic once SHiELD physics diags can be loaded efficiently
# @add_to_diags("physics")
def global_biases_physics(resampled, verification, grid):
    logger.info("Preparing global average biases for physics variables")
    bias_errors = bias(verification, resampled, grid.area, HORIZONTAL_DIMS)

    return _vars_to_diags("bias_global_physics", bias_errors, resampled)


def diurnal_cycles(resampled, verification, grid):

    logger.info("Preparing diurnal cycle diagnostics")

    # TODO: Add in different masked diurnal cycles

    diurnal_verif = calc_ds_diurnal_cycle(verification)
    lon = verification["grid_lont"]
    resampled["grid_lont"] = lon
    diurnal_resampled = calc_ds_diurnal_cycle(resampled)

    diurnal_resampled, diurnal_verif = _add_derived_moisture_diurnal_quantities(
        diurnal_resampled, diurnal_verif
    )

    # Add to diagnostics
    verif_diags = _vars_to_diags("verif_diurnal", diurnal_verif, verification)
    resampled_diags = _vars_to_diags("run_diurnal", diurnal_resampled, resampled)

    return dict(**verif_diags, **resampled_diags)


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
    input_data = {}
    input_data["dycore"] = load_diags.load_dycore(args.url, args.grid_spec, catalog)
    input_data["physics"] = load_diags.load_physics(args.url, args.grid_spec, catalog)

    # add derived variables
    input_data = compute_all_derived_vars(input_data)

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
