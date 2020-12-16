import pathlib
import re
from datetime import datetime, timedelta
from typing import List, Union
from functools import singledispatch

import cftime
import numpy as np
import xarray as xr

# SpencerC added this function, it is not public API, but we need it
from xarray.core.resample_cftime import exact_cftime_datetime_difference

from vcm.cubedsphere.constants import TIME_FMT

TOP_LEVEL_DIR = pathlib.Path(__file__).parent.parent.absolute()


@singledispatch
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


@round_time.register
def _round_time_numpy(time: np.ndarray) -> np.ndarray:
    return np.vectorize(round_time)(time)


@round_time.register
def _round_time_xarray(time: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.vectorize(round_time), time)


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
def _cast_datetime_to_datetime(time: datetime) -> datetime:
    return time


@cast_to_datetime.register
def _cast_numpytime_to_datetime(time: np.datetime64):  # type: ignore
    # https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    seconds_since_epoch = (time - unix_epoch) / one_second
    return datetime.utcfromtimestamp(seconds_since_epoch)


@cast_to_datetime.register
def _(time: str):
    return cast_to_datetime(parse_datetime_from_str(parse_timestep_str_from_path(time)))


def convert_timestamps(coord: xr.DataArray) -> xr.DataArray:
    parser = np.vectorize(parse_datetime_from_str)
    return xr.DataArray(parser(coord), dims=coord.dims, attrs=coord.attrs)


def shift_timestamp(time: str, seconds: Union[int, float]) -> str:
    """Add an offset in seconds to a timestamp in YYYYMMDD.HHMMSS format"""
    offset = timedelta(seconds=seconds)
    offset_datetime = parse_datetime_from_str(time) + offset
    return offset_datetime.strftime("%Y%m%d.%H%M%S")


def get_root():
    """Returns the absolute path to the root directory for any machine"""
    return str(TOP_LEVEL_DIR)
