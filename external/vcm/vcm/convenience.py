import os
import pathlib
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Union
from functools import singledispatch

import cftime
import intake
import numpy as np
import xarray as xr
import yaml

# SpencerC added this function, it is not public API, but we need it
from xarray.core.resample_cftime import exact_cftime_datetime_difference

from vcm.cloud import gsutil
from vcm.cloud.remote_data import open_gfdl_data_with_2d
from vcm.cubedsphere.constants import TIME_FMT

TOP_LEVEL_DIR = pathlib.Path(__file__).parent.parent.absolute()


def round_time(t, to=timedelta(seconds=1)):
    """ cftime will introduces noise when decoding values into date objects.
    This rounds time in the date object to the nearest second, assuming the init time
    is at most 1 sec away from a round minute. This is used when merging datasets so
    their time dims match up.

    Args:
        t: datetime or cftime object
        to: size of increment to round off to. By default round to closest integer
            second.

    Returns:
        datetime or cftime object rounded to nearest minute
    """
    midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)

    time_since_midnight = exact_cftime_datetime_difference(midnight, t)
    remainder = time_since_midnight % to
    quotient = time_since_midnight // to
    if remainder <= to / 2:
        closest_multiple_of_to = quotient
    else:
        closest_multiple_of_to = quotient + 1

    rounded_time_since_midnight = closest_multiple_of_to * to

    return midnight + rounded_time_since_midnight


def encode_time(time: cftime.DatetimeJulian) -> str:
    return time.strftime(TIME_FMT)


def parse_timestep_str_from_path(path: str) -> str:
    """
    Get the model timestep timestamp from a given path

    Args:
        path: A file or directory path that includes a timestep to extract

    Returns:
        The extrancted timestep string
    """

    extracted_time = re.search(r"(\d\d\d\d\d\d\d\d\.\d\d\d\d\d\d)", path)

    if extracted_time is not None:
        return extracted_time.group(1)
    else:
        raise ValueError(f"No matching time pattern found in path: {path}")


def parse_datetime_from_str(time: str) -> cftime.DatetimeJulian:
    """
    Retrieve a datetime object from an FV3GFS timestamp string
    """
    t = datetime.strptime(time, TIME_FMT)
    return cftime.DatetimeJulian(t.year, t.month, t.day, t.hour, t.minute, t.second)


def parse_current_date_from_str(time: str) -> List[int]:
    """Retrieve the 'current_date' in the format required by fv3gfs namelist
    from timestamp string."""
    t = parse_datetime_from_str(time)
    return [t.year, t.month, t.day, t.hour, t.minute, t.second]


# use typehints to dispatch to overloaded datetime casting function
@singledispatch
def cast_to_datetime(
    time: Union[datetime, cftime.DatetimeJulian, np.datetime64]
) -> datetime:
    """Cast datetime-like object to python datetime. Assumes calendars are
    compatible."""
    return datetime(
        time.year,
        time.month,
        time.day,
        time.hour,
        time.minute,
        time.second,
        time.microsecond,
    )


@cast_to_datetime.register
def _(time: datetime) -> datetime:
    return time


@cast_to_datetime.register
def _(time: np.datetime64):
    # https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (time - unix_epoch) / one_second
    return datetime.utcfromtimestamp(seconds_since_epoch)


def convert_timestamps(coord: xr.DataArray) -> xr.DataArray:
    parser = np.vectorize(parse_datetime_from_str)
    return xr.DataArray(parser(coord), dims=coord.dims, attrs=coord.attrs)


def get_root():
    """Returns the absolute path to the root directory for any machine"""
    return str(TOP_LEVEL_DIR)


def get_shortened_dataset_tags():
    """"Return a dictionary mapping short dataset definitions to the full names"""
    short_dset_yaml = (
        pathlib.Path(TOP_LEVEL_DIR, "configurations") / "short_datatag_defs.yml"
    )
    return yaml.load(short_dset_yaml.open(), Loader=yaml.SafeLoader)


root = get_root()


# TODO: I believe fv3_data_root and paths are legacy
fv3_data_root = "/home/noahb/data/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute/"

paths = {
    "1deg_src": f"{root}/data/interim/advection/"
    "2019-07-17-FV3_DYAMOND_0.25deg_15minute_regrid_1degree.zarr"
}


def open_catalog():
    return intake.Catalog(f"{root}/catalog.yml")


def open_dataset(tag) -> xr.Dataset:
    short_dset_tags = get_shortened_dataset_tags()
    if tag == "gfdl":
        return open_gfdl_data_with_2d(open_catalog())
    elif tag == "1degTrain":
        return (
            open_dataset("1deg")
            .merge(xr.open_zarr(paths["1deg_src"]))
            .pipe(_insert_apparent_heating)
        )
    else:
        if tag in short_dset_tags:
            # TODO: Debug logging about tag transformation
            tag = short_dset_tags[tag]

        curr_catalog = open_catalog()
        source = curr_catalog[tag]
        dataset = source.to_dask()

        for transform in source.metadata.get("data_transforms", ()):
            print(f"Applying data transform: {transform}")
            transform_function = globals()[transform]
            dataset = transform_function(dataset)

        return dataset


def open_data(sources=False, two_dimensional=True):
    path = os.path.join(
        root, "data/interim/2019-07-17-FV3_DYAMOND_0.25deg_15minute_256x256_blocks.zarr"
    )
    ds = xr.open_zarr(path)

    if sources:
        src = xr.open_zarr(root + "data/interim/apparent_sources.zarr")
        ds = ds.merge(src)

    if two_dimensional:
        data_2d = xr.open_zarr(fv3_data_root + "2d.zarr")
        ds = xr.merge([ds, data_2d], join="left")

    return ds


# Data Adjustments ##
def _rename_SHiELD_varnames_to_orig(ds: xr.Dataset) -> xr.Dataset:
    """
    Replace varnames from new dataset to match original style
    from initial DYAMOND data.
    """

    rename_list = {
        "ucomp": "u",
        "vcomp": "v",
        "sphum": "qv",
        "HGTsfc": "zs",
        "delz": "dz",
        "delp": "dp",
    }
    return ds.rename(rename_list)


def replace_esmf_coords_reg_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    Replace ESMF coordinates from regridding to a regular lat/lon grid
    """

    lat_1d = ds.lat.isel(x=0).values
    lon_1d = ds.lon.isel(y=0).values
    ds = ds.assign_coords({"x": lon_1d, "y": lat_1d})
    ds = ds.rename({"lat": "lat_grid", "lon": "lon_grid"})

    return ds


def _replace_esmf_coords(ds):
    #  Older version using DYAMOND data -AP Oct 2019
    return (
        # DYAMOND data
        ds.assign_coords(x=ds.lon.isel(y=0), y=ds.lat.isel(x=0))
        .drop(["lat", "lon"])
        .rename({"x": "lon", "y": "lat"})
    )


def _insert_apparent_heating(ds: xr.Dataset) -> xr.Dataset:
    from vcm.calc.calc import apparent_heating

    dqt_dt = ds.advection_qv + ds.storage_qv
    dtemp_dt = ds.advection_temp + ds.storage_temp
    return ds.assign(q1=apparent_heating(dtemp_dt, ds.w), q2=dqt_dt)


def file_names_for_time_step(timestep, category, resolution=3072):
    # TODO remove this hardcode
    bucket = f"gs://vcm-ml-data"
    f"2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/"
    f"C{resolution}/{timestep}/{timestep}.{category}*"
    return gsutil.list_matches(bucket)


def tile_num(name):
    return name[-len("1.nc.0000")]


def group_file_names(files):
    out = defaultdict(list)
    for file in files:
        out[tile_num(file)].append(file)

    return out


def map_ops(fun, grouped_files, *args):
    out = {}
    for key, files in grouped_files.items():
        seq = []
        for file in files:
            ds = fun(file, *args)
            seq.append(ds)
        out[key] = seq
    return out
