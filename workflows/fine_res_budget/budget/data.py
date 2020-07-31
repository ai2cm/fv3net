import datetime
import logging

import fsspec
import numpy as np
import xarray as xr

import vcm
from vcm.convenience import round_time

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


def open_diagnostic_data(url: str) -> xr.Dataset:
    logger.info(f"Open diagnostic data at {url}")
    ds = xr.open_zarr(fsspec.get_mapper(url))
    return standardize_diagnostic_metadata(ds)


def open_atmos_15min_coarse_ave(url: str) -> xr.Dataset:
    return open_diagnostic_data(url)


def open_gfsphysics_15min_coarse(url: str) -> xr.Dataset:
    ds = open_diagnostic_data(url)
    # Like the atmos_15min_coarse_ave dataset, the tendency diagnostics
    # output by the physics are centered in the middle of the output
    # interval; the time coordinate of the physics diagnostics must be
    # corrected to reflect that.
    offset = datetime.timedelta(minutes=-7, seconds=-30)
    return offset_time(ds, offset)


def offset_time(
    ds: xr.Dataset, offset: datetime.timedelta, time_dim: str = "time"
) -> xr.Dataset:
    """Offset the time coordinate of the Dataset by adding a timedelta."""
    corrected_time = ds[time_dim] + offset
    return ds.assign({time_dim: corrected_time})


def open_restart_data(RESTART_ZARR):
    logger.info(f"Opening restart data at {RESTART_ZARR}")
    store = fsspec.get_mapper(RESTART_ZARR)
    restarts = xr.open_zarr(store)
    return standardize_restart_metadata(restarts)


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


def merge(restarts, diagnostics):
    restarts = shift(restarts)
    datasets = [restarts] + diagnostics
    return xr.merge(datasets, join="inner", compat="override").drop_vars(
        GRID_VARIABLES, errors="ignore"
    )
