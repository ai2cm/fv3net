import datetime
from typing import Tuple
import cftime


def translate_time(time: Tuple[int, int, int, int, int, int]) -> cftime.DatetimeJulian:
    year = time[0]
    month = time[1]
    day = time[2]
    hour = time[4]
    min = time[5]
    return cftime.DatetimeJulian(year, month, day, hour, min)


def from_datetime(dt: datetime.datetime) -> cftime.DatetimeJulian:
    return cftime.DatetimeJulian(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )
