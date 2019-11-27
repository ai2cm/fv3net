import os
import pathlib
from typing.io import BinaryIO
from typing import Dict
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from toolz import groupby, merge
from os.path import join
import attr

import cftime
import fsspec
import pandas as pd
import xarray as xr

TIME_FMT = "%Y%m%d.%H%M%S"


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


def set_initial_time(restart_files, time):
    return [
        replace(file, _time=time) if file.is_initial_time else file for file in
        restart_files
    ]


def set_final_time(restart_files, time):
    return [
        replace(file, _time=time) if file.is_final_time else file for file in
        restart_files
    ]


def infer_initial_time(times):
    return times[0] - (times[1] - times[0])


def infer_final_time(times):
    return times[-1] + (times[-1] - times[-2])


def fill_times(restart_files, initial_time=None, final_time=None):
    times = sorted(set(file.time for file in restart_files))
    if initial_time is None:
        initial_time = infer_initial_time(times)
    if final_time is None:
        final_time = infer_final_time(times)
    return set_final_time(set_initial_time(restart_files, initial_time), final_time)


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
    ds = xr.open_dataset(file.open())
    return {key: ds[key].data for key in ds}


def _load_group(files):
    return merge(_load_restart(file) for file in files)


def _open_all_restarts_at_url(url, initial_time=None, final_time=None):
    restarts = list(_restart_files_at_url(url))
    restarts = fill_times(restarts, initial_time, final_time)
    grouped = groupby(lambda x: (x.time, x.tile), restarts)
    opened = {key: _load_group(files) for key, files in grouped.items()}
    return opened


def assign_time_dims(ds: xr.Dataset, dirs: list, dims: dict) -> xr.Dataset:
    """Assign coordinates to appropriate time dimensions, i.e., initialization
    time and forecast time, and drop uninformative time dimension labels

    """
    if dirs == BOTH_DIRS:
        ds = ds.assign_coords(
            initialization_time=dims["initialization_time"],
            forecast_time=dims["forecast_time"],
        )
    elif dirs == ["RESTART"]:
        ds = ds.assign_coords(
            initialization_time=dims["initialization_time"],
            forecast_time=[dims["forecast_time"][-1]],
        )
    if "Time" in ds.dims:
        ds = ds.squeeze(dim="Time")
    if "Time" in ds.coords:
        ds = ds.drop(labels="Time")
    if "time" in ds.coords:
        ds = ds.drop(labels="time")
    return ds


def open_oro_data(ds) -> xr.Dataset:
    """Open orography files via xr.concat since they are indexed differently than
    other files

    """
    # drop since these are duplicated in the grid spec
    labels_to_drop = ["slmsk", "geolon", "geolat"]
    return ds.drop(labels=labels_to_drop)


def add_vertical_coords(ds: xr.Dataset, run_dir: str, dims: dict) -> xr.Dataset:
    """
    Add the ak and bk 1-D arrays of length phalf stored in fv_core.res.nc to the dataset
    """
    vertical_coords_ds = (
        xr.open_dataset(join(run_dir, "INPUT/fv_core.res.nc"))
        .rename({"xaxis_1": "phalf"})
        .squeeze()
        .drop(labels="Time")
    )
    return xr.merge([ds, vertical_coords_ds])


def use_diagnostic_coordinates(
    ds: xr.Dataset, category: str, output_mapping: dict
) -> xr.Dataset:
    """Map the coordinate names to diagnostic standards using an assumed order for
    dimensions for each file category and variable

    """
    data_vars = {}
    for var in ds.data_vars:
        data_vars[var] = (output_mapping[category][var], ds[var].data)
    return xr.Dataset(data_vars)

