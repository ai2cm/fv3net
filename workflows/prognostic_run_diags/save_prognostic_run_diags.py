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

from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict

import vcm
import load_diagnostic_data as load_diags

import logging

logger = logging.getLogger(__file__)

_DIAG_FNS = defaultdict(list)

HORIZONTAL_DIMS = ["grid_xt", "grid_yt", "tile"]


def add_to_diags(diags_key):
    """
    Add a function to the list of diagnostics to be computed
    for a specified group.

    Args:
        diags_key: A key for a group of diagnostics
        func: a function which computes a set of diagnostics.
            It needs to have the following signature::

                func(resampled_prog_run_data, verificiation_c48, grid)

            and should return an xarray Dataset of diagnostics.
            This output will be merged with all other decoratored functions,
            so some care must be taken to avoid variable and coordinate clashes.
    """

    def wrap(func):
        _DIAG_FNS[diags_key].append(func)
        return func
    return wrap


def compute_all_diagnostics(input_datasets: Dict[str, Tuple[xr.Dataset]]):
    """
    Compute all diagnostics for input data.

    Args
    ----
    input_datasets:
        A key value pair where the key relates to the target diagnostics key
        in _DIAG_FNS, and the value is a tuple of datasets to be provided as
        positional arguments to the diagnostic functions.
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


def rms(x, y, w, dims):
    return np.sqrt(((x - y) ** 2 * w).sum(dims) / w.sum(dims))


def bias(truth, prediction, w, dims):
    return ((truth - prediction) * w).sum(dims) / w.sum(dims)


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

    ds_verif[f"total_P_{filter_flag}"] = ds_verif["PRATEsfc"].assign_attrs({
        "long_name": "diurnal precipitation from verification",
        "units": "kg/m^2/s"
    })
    
    total_P = ds_run["PRATEsfc"]
    if "net_moistening" in ds_run:
        total_P = total_P - ds_run["net_moistening"]  # P - dQ2
    ds_run[f"total_P_{filter_flag}"] = total_P.assign_attrs({
        "long_name": "diurnal precipitation from coarse model run",
        "units": "kg/m^2/s"
    })

    # TODO: add thermo function into top-level import?
    E_run = vcm.calc.thermo.latent_heat_flux_to_evaporation(ds_run["LHTFLsfc"])
    E_verif = vcm.calc.thermo.latent_heat_flux_to_evaporation(ds_verif["LHTFLsfc"])
    E_diff = E_run - E_verif
    P_diff = ds_run[f"total_P_{filter_flag}"] - ds_verif[f"total_P_{filter_flag}"]

    ds_run[f"evap_diff_from_verif_{filter_flag}"] = E_diff.assign_attrs({
        "long_name": "diurnal diff: evap_coarse - evap_hires",
        "units": "kg/m^2/s"
    })
    ds_run[f"precip_diff_from_verif_{filter_flag}"] = P_diff.assign_attrs({
        "long_name": "diurnal diff: precip_coarse - precip_hires",
        "units": "kg/m^2/s"
    })

    net_precip_diff = (P_diff - E_diff)
    ds_run[f"net_precip_diff_from_verif_{filter_flag}"] = net_precip_diff.assign_attrs({
        "long_name": "diurnal diff: net_precip_coarse - net_precip_hires",
        "units": "kg/m^2/s"
    })

    return ds_run, ds_verif


def dump_nc(ds: xr.Dataset, f):
    # to_netcdf closes file, which will delete the buffer
    # need to use a buffer since seek doesn't work with GCSFS file objects
    with tempfile.TemporaryDirectory() as dirname:
        url = os.path.join(dirname, "tmp.nc")
        ds.to_netcdf(url, engine="h5netcdf")
        with open(url, "rb") as tmp1:
            shutil.copyfileobj(tmp1, f)


def _vars_to_diags(suffix, ds, src_ds):
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


@add_to_diags("3H")
def rms_errors(resampled, verification_c48, grid):
    logger.info("Preparing rms errors")
    rms_errors = rms(resampled, verification_c48, grid.area, dims=HORIZONTAL_DIMS)

    return _vars_to_diags("rms_global", rms_errors, resampled)


@add_to_diags("3H")
def global_averages(resampled, verification, grid):
    logger.info("Preparing global averages")
    area_averages = (resampled * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
        HORIZONTAL_DIMS
    )

    return _vars_to_diags("global_avg", area_averages, resampled)


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


def open_tiles(path):
    return xr.open_mfdataset(path + ".tile?.nc", concat_dim="tile", combine="nested")


def load_data_3H(url, grid_spec, catalog):
    logger.info(f"Processing 3H data from run directory at {url}")

    # open grid
    logger.info("Opening Grid Spec")
    grid_c384 = open_tiles(grid_spec)

    # open verification
    logger.info("Opening verification data")
    verification = catalog["40day_c384_atmos_8xdaily"].to_dask()
    verification = verification.merge(grid_c384)

    # block average data
    logger.info("Block averaging the verification data")
    verification_c48 = vcm.cubedsphere.weighted_block_average(
        verification, verification.area, 8, x_dim="grid_xt", y_dim="grid_yt"
    )

    # open data
    atmos_diag_url = os.path.join(url, "atmos_dt_atmos")
    logger.info(f"Opening diagnostic data at {atmos_diag_url}")
    ds = open_tiles(atmos_diag_url).load()
    resampled = ds.resample(time="3H", label="right").nearest()

    verification_c48 = verification_c48.sel(
        time=resampled.time[:-1]
    )  # don't use last time point. there is some trouble

    return resampled, verification_c48, verification_c48[["area"]]


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
    resampled, verification, grid = load_data_3H(args.url, args.grid_spec, catalog)
    input_data["3H"] = (resampled, verification, grid)

    # Data loaded at original time resolution
    verif_base = load_diags.load_verification(["40day_c384_diags_time_avg"], catalog, 8, grid.area)
    data_base = load_diags.load_diagnostics(args.url)
    input_data["base"] = (data_base, verif_base, grid)

    # begin constructing diags
    diags = {}

    # maps
    diags["pwat_run_initial"] = resampled.PWAT.isel(time=0)
    diags["pwat_run_final"] = resampled.PWAT.isel(time=-2)
    diags["pwat_verification_final"] = verification.PWAT.isel(time=-2)

    diags.update(compute_all_diagnostics(input_data))

    # add grid vars
    diags = xr.Dataset(diags, attrs=attrs)
    diags = diags.merge(grid)

    logger.info("Forcing computation.")
    diags = diags.load()

    logger.info(f"Saving data to {args.output}")
    diags.to_netcdf(args.output)
