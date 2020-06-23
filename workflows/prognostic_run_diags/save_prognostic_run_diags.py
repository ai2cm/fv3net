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


@add_to_derived_vars("physics_3H")
def derived_physics_variables_copy(ds: xr.Dataset) -> xr.Dataset:
    return derived_physics_variables(ds)


@add_to_derived_vars("physics_15min")
def derived_physics_variables_copy(ds: xr.Dataset) -> xr.Dataset:
    return derived_physics_variables(ds)


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


def calc_ds_diurnal_cycle(ds):
    """
    Calculates the diurnal cycle on moisture variables.  Expects
    time dimension and longitude variable "lon".
    """
    local_time = vcm.local_time(ds, time="time", lon_var="lon")
    local_time.attrs = {"long_name": "local time", "units": "hour"}

    local_time = np.floor(local_time)  # equivalent to hourly binning
    ds["local_time"] = local_time
    diurnal_ds = ds.groupby("local_time").mean()

    return diurnal_ds


def dump_nc(ds: xr.Dataset, f):
    # to_netcdf closes file, which will delete the buffer
    # need to use a buffer since seek doesn't work with GCSFS file objects
    with tempfile.TemporaryDirectory() as dirname:
        url = os.path.join(dirname, "tmp.nc")
        ds.to_netcdf(url, engine="h5netcdf")
        with open(url, "rb") as tmp1:
            shutil.copyfileobj(tmp1, f)


def _prepare_diag_dict(suffix: str, ds: xr.Dataset, src_ds: xr.Dataset) -> DiagDict:
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

    return _prepare_diag_dict("rms_global", rms_errors, resampled)


@add_to_diags("dycore")
def global_averages_dycore(resampled, verification, grid):
    logger.info("Preparing global averages for dycore variables")
    area_averages = (resampled * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
        HORIZONTAL_DIMS
    )

    return _prepare_diag_dict("global_avg", area_averages, resampled)


@add_to_diags("physics_3H")
def global_averages_physics(resampled, verification, grid):
    logger.info("Preparing global averages for physics variables")
    area_averages = (resampled * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
        HORIZONTAL_DIMS
    )

    return _prepare_diag_dict("global_phys_avg", area_averages, resampled)


# TODO: enable this diagnostic once SHiELD physics diags can be loaded efficiently
# @add_to_diags("physics_3H")
def global_biases_physics(resampled, verification, grid):
    logger.info("Preparing global average biases for physics variables")
    bias_errors = bias(verification, resampled, grid.area, HORIZONTAL_DIMS)

    return _prepare_diag_dict("bias_global_physics", bias_errors, resampled)


@add_to_diags("physics_15min")
def diurnal_cycles(resampled, verification, grid):

    logger.info("Preparing diurnal cycle diagnostics")

    diurnal_cycle_vars = [
        f"column_integrated_{a}Q{b}" for a in ["d", "p", ""] for b in ["1", "2"]
    ] + ["SLMSKsfc"]
    resampled = resampled[[var for var in diurnal_cycle_vars if var in resampled]]

    resampled["lon"] = grid["lon"]
    diag_dicts = {}
    for surface_name in ["global", "land", "sea", "seaice"]:
        masked = vcm.mask_to_surface_type(
            resampled, surface_name, surface_type_var="SLMSKsfc"
        )
        diurnal_ds = calc_ds_diurnal_cycle(masked)
        diag_dicts.update(
            _prepare_diag_dict(f"diurnal_{surface_name}", diurnal_ds, resampled)
        )
    return diag_dicts


def _catalog():
    TOP_LEVEL_DIR = Path(os.path.abspath(__file__)).parent.parent.parent
    return str(TOP_LEVEL_DIR / "catalog.yml")


def _resample_time(
    arg: DiagArg, freq_label: str, time_slice=slice(None, -1)
) -> DiagArg:
    prognostic, verification, grid = arg
    prognostic = prognostic.resample(time=freq_label, label="right").nearest()
    prognostic = prognostic.isel(time=time_slice)
    if "time" in verification:  # verification might be an empty dataset
        verification = verification.sel(time=prognostic.time)
    return prognostic, verification, grid


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

    resampled_data = {
        "dycore": _resample_time(loaded_data["dycore"], "3H"),
        "physics_3H": _resample_time(loaded_data["physics"], "3H"),
        "physics_15min": _resample_time(loaded_data["physics"], "15min"),
    }

    # add derived variables
    input_data = compute_all_derived_vars(resampled_data)

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
