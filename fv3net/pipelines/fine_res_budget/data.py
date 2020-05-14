"""
>>> from merge_restarts_and_diags import *
>>> restart_url = "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr/"
>>> url = "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/"

>>> restarts = open_restart_data(restart_url)
>>> diag = open_diagnostic_output(url)
"""  # noqa
import datetime
import logging

import fsspec
import numpy as np
import xarray as xr
import zarr

import vcm
from vcm.convenience import round_time

logger = logging.getLogger(__file__)


def remove_coarse_name(ds):
    name_dict = {}
    for variable in ds:
        suffix = "_coarse"
        if variable.endswith(suffix):
            name_dict[variable] = variable[: -len(suffix)]
    return ds.rename(name_dict)


def rename_dims(ds):
    name_dict = {}
    for variable in ds.dims:
        suffix = "_coarse"
        if variable.endswith(suffix):
            name_dict[variable] = variable[: -len(suffix)]
    return ds.rename_dims(name_dict).rename(name_dict)


def rename_latlon(ds):
    return ds.rename(
        {"grid_lat": "latb", "grid_lon": "lonb", "grid_lont": "lon", "grid_latt": "lat"}
    )


def open_diagnostic_output(url):
    logger.info(f"Opening Diagnostic data at {url}")
    # open diagnostic output
    ds = xr.open_zarr(fsspec.get_mapper(url), consolidated=True)
    # TODO need to implement a correct version of round time
    times = np.vectorize(round_time)(ds.time)
    return (
        ds.assign(time=times)
        .pipe(remove_coarse_name)
        .pipe(rename_dims)
        .pipe(rename_latlon)
    )


def open_restart_data(RESTART_ZARR):
    logger.info(f"Opening restart data at {RESTART_ZARR}")
    coords_names = ["grid_x", "grid_y", "grid_xt", "grid_yt", "pfull", "tile"]
    store = fsspec.get_mapper(RESTART_ZARR)

    if ".zmetadata" not in store:
        zarr.consolidate_metadata(store)

    restarts = xr.open_zarr(store, consolidated=True)
    times = np.vectorize(vcm.parse_datetime_from_str)(restarts.time)
    return restarts.assign(time=times).drop(coords_names)


def standardize_restart_metadata(restarts):
    coords_names = ["grid_x", "grid_y", "grid_xt", "grid_yt", "pfull", "tile"]
    times = np.vectorize(vcm.parse_datetime_from_str)(restarts.time)
    return restarts.assign(time=times).drop(coords_names)


def standardize_diagnostic_metadata(ds):
    times = np.vectorize(round_time)(ds.time)
    return (
        ds.assign(time=times)
        .pipe(remove_coarse_name)
        .pipe(rename_dims)
        .pipe(rename_latlon)
    )


def shift(restarts, dt=datetime.timedelta(seconds=30, minutes=7)):
    time = restarts.time
    begin = restarts.assign(time=time + dt)
    end = restarts.assign(time=time - dt)

    return xr.concat(
        [begin, (begin + end) / 2, end],
        dim=xr.IndexVariable("step", ["begin", "middle", "end"]),
    )


def open_merged_data(restart_url, atmos_coarse_ave_url):
    restarts = open_restart_data(restart_url)
    diag = open_diagnostic_output(atmos_coarse_ave_url)

    restarts = shift(restarts)

    return xr.merge([diag.drop("delp"), restarts], join="inner").drop(
        ["grid_x", "grid_xt", "grid_yt", "grid_y"], errors="ignore"
    )


def merge(restarts, diag):
    restarts = shift(restarts)

    return xr.merge([diag.drop("delp"), restarts], join="inner").drop(
        ["grid_x", "grid_xt", "grid_yt", "grid_y"], errors="ignore"
    )
