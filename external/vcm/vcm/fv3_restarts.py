import os
import re
from collections import defaultdict
from datetime import datetime
from functools import partial
from os.path import join
from typing import Any, Dict, Generator, Tuple

import cftime
import fsspec
import xarray as xr

import f90nml
from vcm.merging import combine_array_sequence

TIME_FMT = "%Y%m%d.%H%M%S"

NUM_SOIL_LAYERS = 4
CATEGORIES = ["fv_core.res", "sfc_data", "fv_tracer", "fv_srf_wnd.res"]
X_NAME = "grid_xt"
Y_NAME = "grid_yt"
X_EDGE_NAME = "grid_x"
Y_EDGE_NAME = "grid_y"
Z_NAME = "pfull"
Z_EDGE_NAME = "phalf"


def _parse_time_string(time):
    t = datetime.strptime(time, TIME_FMT)
    return cftime.DatetimeJulian(t.year, t.month, t.day, t.hour, t.minute, t.second)


def _split_url(url):

    try:
        protocol, path = url.split("://")
    except ValueError:
        protocol = "file"
        path = url

    return protocol, path


def _parse_time(path):
    return re.search(r"(\d\d\d\d\d\d\d\d\.\d\d\d\d\d\d)", path).group(1)


def _get_time(dirname, file, initial_time, final_time):
    if dirname.endswith("INPUT"):
        return initial_time
    elif dirname.endswith("RESTART"):
        try:
            return _parse_time(file)
        except AttributeError:
            return final_time


def _get_tile(file):
    tile = re.search(r"tile(\d)\.nc", file).group(1)
    return int(tile)


def _is_restart_file(file):
    return any(category in file for category in CATEGORIES) and "tile" in file


def _restart_files_at_url(url, initial_time, final_time):
    """List restart files with a given initial and end time within a particular URL

    Yields:
        (time, tile, protocol, path)

    Note:
        the time for the data in INPUT and RESTART cannot be parsed from the file name
        alone so they are required arguments. Some tricky logic such as reading the
        fv_coupler.res file could be done, but I do not think this low-level function
        should have side-effects such as reading a file (which might not always be
        where we expect).

    """
    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if _is_restart_file(file):
                time = _get_time(root, file, initial_time, final_time)
                tile = _get_tile(file)
                yield time, tile, proto, path


def _load_restart(protocol, path):
    fs = fsspec.filesystem(protocol)
    with fs.open(path) as f:
        return xr.open_dataset(f).load()


def _get_grid(rundir):
    proto, path = _split_url(rundir)
    fs = fsspec.filesystem(proto)
    # open namelist
    namelist = join(path, "input.nml")
    with fs.open(namelist, "r") as f:
        s = f.read()
    nml = f90nml.reads(s)["fv_core_nml"]

    # open one file
    return {
        "nz": nml["npz"],
        "nx": nml["npx"] - 1,
        "ny": nml["npy"] - 1,
        "nz_soil": NUM_SOIL_LAYERS,
    }


def _fix_data_array_dimension_names(data_array, nx, ny, nz, nz_soil):
    """Modify dimension names from e.g. xaxis1 to 'x' or 'x_interface' in-place.

    Done based on dimension length (similarly for y).

    Args:
        data_array (DataArray): the object being modified
        nx (int): the number of grid cells along the x-axis
        ny (int): the number of grid cells along the y-axis
        nz (int): the number of grid cells along the z-axis
        nz_soil (int): the number of grid cells along the soil model z-axis

    Returns:
        renamed_array (DataArray): new object with renamed dimensions

    Notes:
        copied from fv3gfs-python
    """
    replacement_dict = {}
    for dim_name, length in zip(data_array.dims, data_array.shape):
        if dim_name[:5] == "xaxis" or dim_name.startswith("lon"):
            if length == nx:
                replacement_dict[dim_name] = X_NAME
            elif length == nx + 1:
                replacement_dict[dim_name] = X_EDGE_NAME
            try:
                replacement_dict[dim_name] = {nx: X_NAME, nx + 1: X_EDGE_NAME}[length]
            except KeyError as e:
                raise ValueError(
                    f"unable to determine dim name for dimension "
                    f"{dim_name} with length {length} (nx={nx})"
                ) from e
        elif dim_name[:5] == "yaxis" or dim_name.startswith("lat"):
            try:
                replacement_dict[dim_name] = {ny: Y_NAME, ny + 1: Y_EDGE_NAME}[length]
            except KeyError as e:
                raise ValueError(
                    f"unable to determine dim name for dimension "
                    f"{dim_name} with length {length} (ny={ny})"
                ) from e
        elif dim_name[:5] == "zaxis":
            try:
                replacement_dict[dim_name] = {nz: Z_NAME, nz_soil: Z_EDGE_NAME}[length]
            except KeyError as e:
                raise ValueError(
                    f"unable to determine dim name for dimension "
                    f"{dim_name} with length {length} (nz={nz})"
                ) from e
    return data_array.rename(replacement_dict).variable


def _fix_metadata(ds, grid):
    try:
        ds_no_time = ds.isel(Time=0).drop("Time")
    except ValueError:
        ds_no_time = ds

    ds_correct_metadata = ds_no_time.apply(
        partial(_fix_data_array_dimension_names, **grid)
    )

    return ds_correct_metadata


def _load_arrays(
    restart_files, grid
) -> Generator[Tuple[Any, Tuple, xr.DataArray], None, None]:
    for (time, tile, protocol, path) in restart_files:
        ds = _load_restart(protocol, path)
        ds_correct_metadata = _fix_metadata(ds, grid)
        time_obj = _parse_time_string(time)
        for var in ds_correct_metadata:
            yield var, (time_obj, tile), ds_correct_metadata[var]


def open_restarts(
    url: str, initial_time: str, final_time: str, grid: Dict[str, int] = None
) -> xr.Dataset:
    """Opens all the restart file within a certain path

    The dimension names are the same as the diagnostic output

    Args:
        url: a URL to the root directory of the. Can be any type of protocol used by
            fsspec, such as google cloud storage 'gs://path-to-rundir'. If no protocol
            prefix is used, then it will be assumed to be a path to a local file
        initial_time: A YYYYMMDD.HHMMSS string for the initial condition. will be parsed
            with CFTime
        final_time: same as `initial_time` but for the final time
        grid: a dict with the grid information (e.g.)::

             {'nz': 79, 'nz_soil': 4, 'nx': 48, 'ny': 48}

    Returns:
        a combined dataset of all the restart files. This is currently not a
        lazy operation. All the data is loaded.

    """
    if grid is None:
        grid = _get_grid(url)
    restart_files = _restart_files_at_url(url, initial_time, final_time)
    arrays = _load_arrays(restart_files, grid)
    return xr.Dataset(combine_array_sequence(arrays, labels=["time", "tile"]))
