#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

import fsspec
import tempfile
import intake
import numpy as np
import xarray as xr
import shutil

import fv3net
import vcm

import logging

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

_METRICS_FNS = []

HORIZONTAL_DIMS = ["grid_xt", "grid_yt", "tile"]


def add_to_metrics(func):
    _METRICS_FNS.append(func)
    return func


def rms(x, y, w, dims):
    return np.sqrt(((x - y) ** 2 * w).sum(dims) / w.sum(dims))


def dump_nc(ds: xr.Dataset, f):
    # to_netcdf closes file, which will delete the buffer
    # need to use a buffer since seek doesn't work with GCSFS file objects
    with tempfile.TemporaryDirectory() as dirname:
        url = os.path.join(dirname, "tmp.nc")
        ds.to_netcdf(url, engine="h5netcdf")
        with open(url, "rb") as tmp1:
            shutil.copyfileobj(tmp1, f)


@add_to_metrics
def rms_errors(resampled, verification_c48, grid):
    rms_errors = rms(
        resampled, verification_c48, grid.area, dims=HORIZONTAL_DIMS
    )

    diags = {}
    for variable in rms_errors:
        lower = variable.lower()
        diags[f"{lower}_rms_global"] = rms_errors[variable].assign_attrs(
            ds[variable].attrs
        )

    return diags


@add_to_metrics
def global_averages(resampled, verification, grid):
    diags = {}
    area_averages = (resampled * resampled.area).sum(HORIZONTAL_DIMS) / resampled.area.sum(
        HORIZONTAL_DIMS
    )
    for variable in area_averages:
        lower = variable.lower()
        diags[f"{lower}_global_avg"] = area_averages[variable].assign_attrs(
            ds[variable].attrs
        )
    return diags


def biases(resampled, verification, grid):




if __name__ == "__main__":

    CATALOG = str(fv3net.TOP_LEVEL_DIR / "catalog.yml")

    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument(
        "--grid-spec",
        default="gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy/grid_spec",
    )
    parser.add_argument("--catalog", default=CATALOG)

    args = parser.parse_args()

    attrs = vars(args)
    attrs["history"] = " ".join(sys.argv)

    url = args.url
    logger.info(f"Processing run directory at {url}")

    # open grid
    logger.info("Opening Grid Spec")
    grid_c384 = vcm.open_tiles(args.grid_spec)

    # open verification
    catalog = intake.open_catalog(args.catalog)
    verification = catalog["40day_c384_atmos_8xdaily"].to_dask()
    verification = verification.merge(grid_c384)
    # block average data
    verification_c48 = vcm.cubedsphere.weighted_block_average(
        verification, verification.area, 8, x_dim="grid_xt", y_dim="grid_yt"
    )

    # open data
    atmos_diag_url = os.path.join(url, "atmos_dt_atmos")
    ds = vcm.open_tiles(atmos_diag_url).load()
    resampled = ds.resample(time="3H", label="right").nearest()
    grid_c48 = resampled[vcm.cubedsphere.constants.GRID_VARS]

    verification_c48 = verification_c48.sel(
        time=resampled.time[:-1]
    )  # don't use last time point. there is some trouble

    # begin constructing diags
    diags = {}

    # maps
    diags["pwat_run_initial"] = ds.PWAT.isel(time=0)
    diags["pwat_run_final"] = ds.PWAT.isel(time=-2)
    diags["pwat_verification_final"] = verification_c48.PWAT.isel(time=-2)

    for metrics_fn in _METRICS_FNS:
        diags.update(metrics_fn(resampled, verification_c48, grid_c48))

    # add grid vars
    diags = xr.Dataset(diags, attrs=attrs)
    diags = diags.merge(grid_c48)

    logger.info(f"Saving data to {args.output}")
    with fsspec.open(args.output, mode="wb") as f:
        dump_nc(diags, f)
