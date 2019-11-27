import os
import pathlib
from typing.io import BinaryIO
from typing import Dict
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from os.path import join
import attr

import cftime
import fsspec
import pandas as pd
import xarray as xr

data_id = "data/raw/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute"
output_2d = "data/interim/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute_2d.zarr"
output_3d = "data/interim/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute_3d.zarr"

files_3d = [
    "h_plev_C3072_1536x768.fre.nc",
    "qi_plev_C3072_1536x768.fre.nc",
    "ql_plev_C3072_1536x768.fre.nc",
    "q_plev_C3072_1536x768.fre.nc",
    "t_plev_C3072_1536x768.fre.nc",
    "u_plev_C3072_1536x768.fre.nc",
    "v_plev_C3072_1536x768.fre.nc",
    "pres_C3072_1536x768.fre.nc",
    "qi_C3072_1536x768.fre.nc",
    "ql_C3072_1536x768.fre.nc",
    "qr_C3072_1536x768.fre.nc",
    "qv_C3072_1536x768.fre.nc",
    "temp_C3072_1536x768.fre.nc",
    "w_C3072_1536x768.fre.nc",
    "u_C3072_1536x768.fre.nc",
    "v_C3072_1536x768.fre.nc",
]

files_2d = [
    "cape_C3072_1536x768.fre.nc",
    "cin_C3072_1536x768.fre.nc",
    "cldc_C3072_1536x768.fre.nc",
    "flds_C3072_1536x768.fre.nc",
    "flus_C3072_1536x768.fre.nc",
    "flut_C3072_1536x768.fre.nc",
    "fsds_C3072_1536x768.fre.nc",
    "fsdt_C3072_1536x768.fre.nc",
    "fsus_C3072_1536x768.fre.nc",
    "fsut_C3072_1536x768.fre.nc",
    "h500_C3072_1536x768.fre.nc",
    "intqg_C3072_1536x768.fre.nc",
    "intqi_C3072_1536x768.fre.nc",
    "intql_C3072_1536x768.fre.nc",
    "intqr_C3072_1536x768.fre.nc",
    "intqs_C3072_1536x768.fre.nc",
    "intqv_C3072_1536x768.fre.nc",
    "lhflx_C3072_1536x768.fre.nc",
    "pr_C3072_1536x768.fre.nc",
    "ps_C3072_1536x768.fre.nc",
    "q2m_C3072_1536x768.fre.nc",
    "rh500_C3072_1536x768.fre.nc",
    "rh700_C3072_1536x768.fre.nc",
    "rh850_C3072_1536x768.fre.nc",
    "shflx_C3072_1536x768.fre.nc",
    "t2m_C3072_1536x768.fre.nc",
    "ts_C3072_1536x768.fre.nc",
    "u10m_C3072_1536x768.fre.nc",
    "u200_C3072_1536x768.fre.nc",
    "ustrs_C3072_1536x768.fre.nc",
    "v10m_C3072_1536x768.fre.nc",
    "qs_C3072_1536x768.fre.nc",
    "v200_C3072_1536x768.fre.nc",
    "vstrs_C3072_1536x768.fre.nc",
]

BOTH_DIRS = ["INPUT", "RESTART"]

CATEGORY_DIR_MAPPING = {
    "grid_spec": ["."],
    "oro_data": ["INPUT"],
    "fv_core.res": BOTH_DIRS,
    "fv_srf_wnd.res": BOTH_DIRS,
    "fv_tracer.res": BOTH_DIRS,
    "sfc_data": BOTH_DIRS,
    "phy_data": ["RESTART"],
}

RESTART_CATEGORIES = CATEGORY_DIR_MAPPING.keys()

# define assumed coordinate order for each file type and for variables within
# each time type that differ from the default arrangement oro
_oro_data_axes_map = defaultdict(lambda: ("tile", "grid_yt", "grid_xt"))
# fv_core
_fv_core_res_axes_map = defaultdict(
    lambda: (
        "tile",
        "forecast_time",
        "initialization_time",
        "Time",
        "pfull",
        "grid_yt",
        "grid_xt",
    )
)
_fv_core_res_axes_map["phis"] = (
    "tile",
    "forecast_time",
    "initialization_time",
    "Time",
    "grid_yt",
    "grid_xt",
)
_fv_core_res_axes_map["u"] = (
    "tile",
    "forecast_time",
    "initialization_time",
    "Time",
    "pfull",
    "grid_y",
    "grid_xt",
)
_fv_core_res_axes_map["v"] = (
    "tile",
    "forecast_time",
    "initialization_time",
    "Time",
    "pfull",
    "grid_yt",
    "grid_x",
)
# fv_srf_wnd
_fv_srf_wnd_res_axes_map = defaultdict(
    lambda: (
        "tile",
        "forecast_time",
        "initialization_time",
        "Time",
        "grid_yt",
        "grid_xt",
    )
)
# fv_tracer
_fv_tracer_res_axes_map = defaultdict(
    lambda: (
        "forecast_time",
        "tile",
        "initialization_time",
        "Time",
        "pfull",
        "grid_yt",
        "grid_xt",
    )
)
# sfc data
_sfc_data_axes_map = defaultdict(
    lambda: (
        "tile",
        "forecast_time",
        "initialization_time",
        "Time",
        "grid_yt",
        "grid_xt",
    )
)
for soil_var in ["stc", "smc", "slc"]:
    _sfc_data_axes_map[soil_var] = (
        "tile",
        "forecast_time",
        "initialization_time",
        "Time",
        "soil_levels",
        "grid_yt",
        "grid_xt",
    )


# phy data
_phy_data_axes_map = defaultdict(
    lambda: (
        "tile",
        "forecast_time",
        "initialization_time",
        "Time",
        "grid_yt",
        "grid_xt",
    )
)
_phy_data_axes_map["phy_f3d_01"] = (
    "tile",
    "forecast_time",
    "initialization_time",
    "Time",
    "pfull",
    "grid_yt",
    "grid_xt",
)

# rename restart file dimensions according to diagnostics conventions
CATEGORY_AXES_MAP = {
    "oro_data": _oro_data_axes_map,
    "fv_core.res": _fv_core_res_axes_map,
    "fv_srf_wnd.res": _fv_srf_wnd_res_axes_map,
    "fv_tracer.res": _fv_tracer_res_axes_map,
    "sfc_data": _sfc_data_axes_map,
    "phy_data": _phy_data_axes_map,
}

N_TILES = 6
TILE_SUFFIXES = [f".tile{tile}.nc" for tile in range(1, N_TILES + 1)]
TIME_FMT = "%Y%m%d.%H%M%S"
TIMESTEP_LENGTH_MINUTES = 15


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


def restart_files_at_url(url):
    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            path = os.path.join(root, file)
            try:
                yield RestartFile.from_path(path, protocol=proto)
            except:
                pass


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


def open_files(files, **kwargs):
    paths = [join(data_id, path) for path in files]
    return xr.open_mfdataset(paths, **kwargs)


def run_dir_cubed_sphere_filepaths(
    run_dir: str, category: str, tile_suffixes: list, target_dirs: list
) -> list:
    """Create nested list of .nc files to open for a given run_dir and restart
    file category

    """
    return [
        [
            [
                join(join(run_dir, target_dir), category + tile_suffix)
                for tile_suffix in tile_suffixes
            ]
            for target_dir in target_dirs
        ]
    ]


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


def combine_file_categories(
    run_dir: str,
    category_mapping: dict,
    tile_suffixes: list,
    new_dims: dict,
    output_mapping: dict,
) -> xr.Dataset:
    """Take a dict of file categories and their mapping to sub_dirs in a run_dir,
    and then opens and stitch together the input and restart files and renaming
    coordinates to match fv3gvs diagnostic output, and timestepping goals,
    i.e., having both initialization time and forecast time

    """
    ds_dict = {}
    for category, dirs in category_mapping.items():
        paths = run_dir_cubed_sphere_filepaths(
            run_dir=run_dir,
            category=category,
            tile_suffixes=tile_suffixes,
            target_dirs=dirs,
        )
        if category == "oro_data":
            ds = open_oro_data(paths)
        elif category == "fv_tracer.res":
            ds = open_fv_tracer(paths)
        else:
            ds = xr.open_mfdataset(
                paths=paths,
                concat_dim=["initialization_time", "forecast_time", "tile"],
                combine="nested",
            )
        if category == "grid_spec":
            ds = ds.squeeze()
        else:
            ds = use_diagnostic_coordinates(ds, category, output_mapping)
        if "tile" in new_dims:
            ds = ds.assign_coords(tile=new_dims["tile"])
        ds = assign_time_dims(ds, dirs, new_dims)
        ds_dict[category] = ds
    ds_merged = add_vertical_coords(
        ds=xr.merge([ds for ds in ds_dict.values()]), run_dir=run_dir, dims=new_dims
    )
    return ds_merged.assign_coords(
        pfull=range(1, ds_merged.dims["pfull"] + 1),
        soil_levels=range(1, ds_merged.dims["soil_levels"] + 1),
    )


def rundir_to_dataset(rundir: str, initial_timestep: str) -> xr.Dataset:
    tile = pd.Index(range(1, N_TILES + 1), name="tile")
    t = datetime.strptime(initial_timestep, TIME_FMT)
    initialization_time = [
        cftime.DatetimeJulian(t.year, t.month, t.day, t.hour, t.minute, t.second)
    ]
    # TODO: add functionality for passing timestep length and number of restart
    # timesteps
    forecast_time = [timedelta(minutes=0), timedelta(minutes=TIMESTEP_LENGTH_MINUTES)]
    new_dims = {
        "initialization_time": initialization_time,
        "forecast_time": forecast_time,
        "tile": tile,
    }
    ds = combine_file_categories(
        run_dir=rundir,
        category_mapping=CATEGORY_DIR_MAPPING,
        new_dims=new_dims,
        output_mapping=CATEGORY_AXES_MAP,
        tile_suffixes=TILE_SUFFIXES,
    )
    return ds



def main():
    ds_3d = open_files(files_3d)
    ds_3d.to_zarr(output_3d, mode="w")

    ds_2d = open_files(files_2d)
    ds_2d.to_zarr(output_2d, mode="w")


if __name__ == "__main__":
    main()
