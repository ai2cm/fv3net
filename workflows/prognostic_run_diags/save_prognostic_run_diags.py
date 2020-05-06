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
import warnings

from pathlib import Path
from datetime import timedelta

import vcm

import logging

logger = logging.getLogger(__file__)

_DIAG_FNS = []

HORIZONTAL_DIMS = ["grid_xt", "grid_yt", "tile"]


def add_to_diags(func):
    """Add a function to the list of diagnostics to be computed

    Args:
        func: a function which computes a set of diagnostics.
        It needs to have the following signature::

            func(resampled_prog_run_data, verificiation_c48, grid)

        and should return an xarray Dataset of diagnostics.
        This output will be merged with all other decoratored functions,
        so some care must be taken to avoid variable and coordinate clashes.
    """
    _DIAG_FNS.append(func)
    return func


def compute_all_diagnostics(resampled, verification, grid):
    diags = {}
    logger.info("Computing all metrics")
    for metrics_fn in _DIAG_FNS:
        logger.info(f"Computing {metrics_fn}")
        current_diags = metrics_fn(resampled, verification, grid)
        _warn_on_overlap(diags, current_diags)
        diags.update(current_diags)
    return diags


def _warn_on_overlap(old, new):
    overlap = set(old) & set(new)
    if len(overlap) > 0:
        warnings.warn(
            UserWarning(
                f"Overlapping keys detected: {overlap}. Updates will overwrite "
                "pre-existing keys."
            )
        )


def rms(x, y, w, dims):
    return np.sqrt(((x - y) ** 2 * w).sum(dims) / w.sum(dims))


def bias(truth, prediction, w, dims):
    return ((truth - prediction) * w).sum(dims) / w.sum(dims)


def dump_nc(ds: xr.Dataset, f):
    # to_netcdf closes file, which will delete the buffer
    # need to use a buffer since seek doesn't work with GCSFS file objects
    with tempfile.TemporaryDirectory() as dirname:
        url = os.path.join(dirname, "tmp.nc")
        ds.to_netcdf(url, engine="h5netcdf")
        with open(url, "rb") as tmp1:
            shutil.copyfileobj(tmp1, f)


@add_to_diags
def rms_errors(resampled, verification_c48, grid):
    rms_errors = rms(resampled, verification_c48, grid.area, dims=HORIZONTAL_DIMS)

    diags = {}
    for variable in rms_errors:
        lower = variable.lower()
        diags[f"{lower}_rms_global"] = rms_errors[variable].assign_attrs(
            resampled[variable].attrs
        )

    return diags


@add_to_diags
def global_averages(resampled, verification, grid):
    diags = {}
    area_averages = (resampled * grid.area).sum(HORIZONTAL_DIMS) / grid.area.sum(
        HORIZONTAL_DIMS
    )
    for variable in area_averages:
        lower = variable.lower()
        diags[f"{lower}_global_avg"] = area_averages[variable].assign_attrs(
            resampled[variable].attrs
        )
    return diags


@add_to_diags
def diurnal_cycle(resampled, verification, grid):
    moist_vars = ["LHTFLsfc", "PRATEsfc"]


def open_tiles(path):
    return xr.open_mfdataset(path + ".tile?.nc", concat_dim="tile", combine="nested")


def _round_microseconds(dt):
    inc = timedelta(seconds=round(dt.microsecond * 1e-6))
    dt = dt.replace(microsecond=0)
    dt += inc
    return dt


def _round_time_coord(ds, time_coord="time"):
    
    new_times = np.array(list(map(_round_microseconds, ds.time.values)))
    ds = ds.assign_coords({time_coord: new_times})
    return ds


def _remove_name_suffix(ds, target):
    replace_names = {vname: vname.replace(target, "")
                     for vname in ds if target in vname}
    replace_names.update(
        {dim: dim.replace(target, "")
         for dim in ds.dims if target in dim}
    )
    ds = ds.rename(replace_names)
    return ds


def _join_sfc_zarr_coords(sfc_nc, sfc_zarr):
    sfc_nc = sfc_nc.assign_coords({"tile": np.arange(6)})
    
    # If prog run crashes there's one less
    if len(sfc_nc.time) < len(sfc_zarr.time):
        sfc_zarr = sfc_zarr.isel(time=slice(0, -1))

    # TODO: should adjust how the zarr is saved to make this easier
    sfc_zarr = sfc_zarr.rename({"x": "grid_xt", "y": "grid_yt", "rank": "tile"})
    sfc_zarr = sfc_zarr.assign_coords(sfc_nc.coords)

    joined = sfc_nc.merge(sfc_zarr, join="inner")

    return joined


def load_data(url, grid_spec, catalog):
    logger.info(f"Processing run directory at {url}")

    # open grid
    logger.info("Opening Grid Spec")
    grid_c384 = open_tiles(grid_spec)

    # open verification
    logger.info("Opening verification data")
    verification = catalog["40day_c384_atmos_8xdaily"].to_dask()
    verification = verification.merge(grid_c384)
    # load moisture vars for diurnal cycle
    moist_verif = catalog["40day_c384_diags_time_avg"].to_dask()
    moist_verif = _round_time_coord(moist_verif)
    moist_verif = _remove_name_suffix(moist_verif, "_coarse")
    moist_verif = moist_verif.assign_coords({"tile": np.arange(6)})
    moisture_vars = ["PRATEsfc", "LHTFLsfc"]
    moist_verif = moist_verif[moisture_vars]
    verification = verification.merge(moist_verif, join="inner") # merges to 3H
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

    # load sfc diagnostics
    sfc_diag_url = os.path.join(url, "sfc_dt_atmos")
    sfc_diag = open_tiles(sfc_diag_url)
    sfc_diag = sfc_diag.assign_coords({"tile": np.arange(6)})

    sfc_zarr_url = os.path.join(url, "diags.zarr")
    if os.path.exists(sfc_zarr_url):
        sfc_zarr = xr.open_zarr(sfc_zarr_url)
        sfc_diag = _join_sfc_zarr_coords(sfc_diag, sfc_zarr)
        if "net_moistening" in sfc_diag:
            moisture_vars += ["net_moistening"]
    
    sfc_diag = sfc_diag[moisture_vars]
    resampled = resampled.merge(sfc_diag, join="inner")

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
    resampled, verification, grid = load_data(args.url, args.grid_spec, catalog)

    # begin constructing diags
    diags = {}

    # maps
    diags["pwat_run_initial"] = resampled.PWAT.isel(time=0)
    diags["pwat_run_final"] = resampled.PWAT.isel(time=-2)
    diags["pwat_verification_final"] = verification.PWAT.isel(time=-2)

    diags.update(compute_all_diagnostics(resampled, verification, grid))

    # add grid vars
    diags = xr.Dataset(diags, attrs=attrs)
    diags = diags.merge(grid)

    logger.info("Forcing computation.")
    diags = diags.load()

    logger.info(f"Saving data to {args.output}")
    diags.to_netcdf(args.output)
