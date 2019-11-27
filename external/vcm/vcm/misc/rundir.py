import os
import f90nml
from typing.io import BinaryIO
from typing import Dict
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from os.path import join

import cftime
import fsspec
import pandas as pd
import xarray as xr

TIME_FMT = "%Y%m%d.%H%M%S"

NUM_SOIL_LAYERS = 4
CATEGORIES = ['fv_core.res', 'sfc_data', 'fv_tracer', 'fv_srf_wnd.res']
X_NAME = 'grid_xt'
Y_NAME = 'grid_yt'
X_EDGE_NAME = 'grid_x'
Y_EDGE_NAME = 'grid_y'
Z_NAME = 'pfull'
Z_EDGE_NAME = 'phalf'


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


@dataclass
class RestartFile:
    path: str
    category: str
    tile: int
    subtile: int
    protocol: str = 'file'
    _time: str = None


    @classmethod
    def from_path(cls, path, protocol='file'):
        pattern = cls._restart_regexp()
        matches = pattern.search(path)
        time = matches.group("time")
        category = matches.group("category")
        tile = matches.group("tile")
        subtile = matches.group("subtile")
        subtile = None if subtile is None else int(subtile)

        if 'grid' in category:
            raise ValueError(f"Data {category} is not a restart file")

        return cls(path, category=category, tile=int(tile), subtile=subtile,
                   _time=time, protocol=protocol)

    @property
    def time(self):
        try:
            return _parse_time_string(self._time)
        except TypeError:
            return None

    @property
    def is_initial_time(self):
        dirname = os.path.dirname(self.path)
        return dirname.endswith("INPUT")

    @property
    def is_final_time(self):
        in_restart = not self.is_initial_time
        return in_restart and self.time is None

    @staticmethod
    def _restart_regexp():
        pattern = (
            r"(?P<time>\d\d\d\d\d\d\d\d\.\d\d\d\d\d\d)?\.?"
            r"(?P<category>[^\/]*)\."
            r"tile(?P<tile>\d)\.nc"
            r"\.?(?P<subtile>\d\d\d\d)?"
        )

        return re.compile(pattern)

    def open(self) -> BinaryIO:
        return fsspec.filesystem(self.protocol).open(self.path, "rb")


def _set_initial_time(restart_files, time):
    return [
        replace(file, _time=time) if file.is_initial_time else file for file in
        restart_files
    ]


def _set_final_time(restart_files, time):
    return [
        replace(file, _time=time) if file.is_final_time else file for file in
        restart_files
    ]


def _fill_times(restart_files, initial_time=None, final_time=None):
    times = sorted(set(file.time for file in restart_files))
    return _set_final_time(_set_initial_time(restart_files, initial_time), final_time)


def _restart_files_at_url(url):
    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            path = os.path.join(root, file)
            try:
                yield RestartFile.from_path(path, protocol=proto)
            except:
                pass


def _load_restart(file: RestartFile):
    with file.open() as f:
        return xr.open_dataset(f).load()


def _open_all_restarts_at_url(url, initial_time=None, final_time=None):
    restarts = list(_restart_files_at_url(url))
    return _fill_times(restarts, initial_time, final_time)


def open_restarts(url: str, initial_time: str, final_time: str, grid: Dict[str, int]=None) -> xr.Dataset:
    """Opens all the restart file within a certain path

    The dimension names are the same as the diagnostic output

    Args:
        url: a URL to the root directory of the. Can be any type of protocol used by
            fsspec, such as google cloud storage 'gs://path-to-rundir'. If no protocol prefix is used, then it will be
            assumed to be a path to a local file
        initial_time: A YYYYMMDD.HHMMSS string for the initial condition. will be parsed to CFTime
        final_time: same as `initial_time` but for the final time
        grid: a dict with the grid information (e.g.)::

             {'nz': 79, 'nz_soil': 4, 'nx': 48, 'ny': 48}

    Returns:
        a combined dataset of all the restart files. This is currently not a
        lazy operation. All the data is loaded.

    """
    restarts = _open_all_restarts_at_url(url, initial_time, final_time)

    if grid is None:
        grid = _get_grid(url)

    output = defaultdict(dict)
    for restart in restarts:
        ds = _load_restart(restart)
        try:
            ds_no_time = ds.isel(Time=0).drop('Time')
        except ValueError:
            ds_no_time = ds

        ds_correct_metadata = ds_no_time.apply(partial(_fix_data_array_dimension_names, **grid))
        for variable in ds_correct_metadata:
            output[variable][(restart.time, restart.tile)] = ds_correct_metadata[variable]

    return xr.merge([_concat_dict(output[key]) for key in output])


def _get_grid(rundir):
    proto, path = _split_url(rundir)
    fs = fsspec.filesystem(proto)
    # open namelist
    namelist = join(path, 'input.nml')
    with fs.open(namelist, "r") as f:
        s = f.read()
    nml = f90nml.reads(s)['fv_core_nml']

    # open one file
    return {'nz': nml['npz'], 'nx': nml['npx']-1, 'ny': nml['npy'] - 1, 'nz_soil': NUM_SOIL_LAYERS}


def _concat_dict(datasets, dims=['time', 'tile']):
    index = pd.MultiIndex.from_tuples(datasets.keys(), names=dims)
    index.name = 'stacked'
    ds = xr.concat(datasets.values(), dim=index)
    return ds.unstack('stacked')


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
        if dim_name[:5] == 'xaxis' or dim_name.startswith('lon'):
            if length == nx:
                replacement_dict[dim_name] = X_NAME
            elif length == nx + 1:
                replacement_dict[dim_name] = X_EDGE_NAME
            try:
                replacement_dict[dim_name] = {nx: X_NAME, nx + 1: X_EDGE_NAME}[length]
            except KeyError as e:
                raise ValueError(
                    f'unable to determine dim name for dimension '
                    f'{dim_name} with length {length} (nx={nx})'
                ) from e
        elif dim_name[:5] == 'yaxis' or dim_name.startswith('lat'):
            try:
                replacement_dict[dim_name] = {ny: Y_NAME, ny + 1: Y_EDGE_NAME}[length]
            except KeyError as e:
                raise ValueError(
                    f'unable to determine dim name for dimension '
                    f'{dim_name} with length {length} (ny={ny})'
                ) from e
        elif dim_name[:5] == 'zaxis':
            try:
                replacement_dict[dim_name] = {nz: Z_NAME, nz_soil: Z_EDGE_NAME}[length]
            except KeyError as e:
                raise ValueError(
                    f'unable to determine dim name for dimension '
                    f'{dim_name} with length {length} (nz={nz})'
                ) from e
    return data_array.rename(replacement_dict).variable

