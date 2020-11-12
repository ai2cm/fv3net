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

import vcm
from vcm import safe
from vcm.convenience import round_time

from . import config

logger = logging.getLogger(__file__)

GRID_VARIABLES = ["grid_x", "grid_y", "grid_xt", "grid_yt", "pfull", "tile"]


def rename_dims(ds):
    name_dict = {}
    for variable in ds.dims:
        suffix = "_coarse"
        if variable.endswith(suffix):
            name_dict[variable] = variable[: -len(suffix)]
    return ds.rename_dims(name_dict).rename(name_dict)


def rename_latlon(ds):
    return ds.rename(
        {
            "grid_lat_coarse": "latb",
            "grid_lon_coarse": "lonb",
            "grid_lont_coarse": "lon",
            "grid_latt_coarse": "lat",
        }
    )


def open_diagnostic_output(url):
    logger.info(f"Opening Diagnostic data at {url}")
    # open diagnostic output
    ds = xr.open_zarr(fsspec.get_mapper(url), consolidated=True)
    return standardize_diagnostic_metadata(ds)


def open_restart_data(RESTART_ZARR):
    logger.info(f"Opening restart data at {RESTART_ZARR}")
    store = fsspec.get_mapper(RESTART_ZARR)
    restarts = xr.open_zarr(store, consolidated=True)
    return standardize_restart_metadata(restarts)


def open_merged(
    restart_url: str = config.restart_url,
    physics_url: str = config.physics_url,
    gfsphysics_url: str = config.gfsphysics_url,
    area_url: str = config.area_url,
) -> xr.Dataset:
    restarts = open_restart_data(restart_url)
    diag = open_diagnostic_output(physics_url)
    gfsphysics = safe.get_variables(
        open_diagnostic_output(gfsphysics_url), config.GFSPHYSICS_VARIABLES
    )
    area_diags = open_diagnostic_output(area_url)

    shifted_restarts = shift(restarts)
    shift_gfs = shift(gfsphysics)

    merged = xr.merge(
        [
            safe.get_variables(shifted_restarts, config.RESTART_VARIABLES),
            safe.get_variables(shift_gfs, config.GFSPHYSICS_VARIABLES).drop("tile"),
            safe.get_variables(diag, config.PHYSICS_VARIABLES),
            area_diags["exposed_area_coarse"],
        ],
        join="inner",
        compat="override",
    ).drop_vars(GRID_VARIABLES, errors="ignore")

    num_tiles = len(merged.tile)
    tiles = range(1, num_tiles + 1)
    return merged.assign_coords(tile=tiles)


def standardize_restart_metadata(restarts):
    times = np.vectorize(vcm.parse_datetime_from_str)(restarts.time)
    return restarts.assign(time=times).drop_vars(GRID_VARIABLES)


def standardize_diagnostic_metadata(ds):
    times = np.vectorize(round_time)(ds.time)
    return ds.assign(time=times).pipe(rename_dims).pipe(rename_latlon)


def shift(restarts, dt=datetime.timedelta(seconds=30, minutes=7)):
    """Define the restart at the center of the time interval

    Here schematic of the time coordinate in model output::

        x-------o--------x-------o-------x
        r1---------------r2-------------r3

    The restart data (r?) are defined at the edges of time intervals ("x"),
    but sometimes we want to do computations with them at the centers of these
    time intervals ("o")::

        x-------o--------x-------o-------x
        -------r1.5------------r2.5-------
    
    ``r1.5`` is an xarray dataset containing ``(r1, (r1+r2)/2, r2)``,
    the beginning, middle, and end of the time step.

    """
    time = restarts.time
    begin = restarts.assign(time=time + dt)
    end = restarts.assign(time=time - dt)

    return xr.concat(
        [begin, (begin + end) / 2, end],
        dim=xr.IndexVariable("step", ["begin", "middle", "end"]),
        join="inner",
    )
