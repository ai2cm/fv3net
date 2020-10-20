from typing import List
import datetime
import cftime

TIME_FMT = "%Y%m%d.%H%M%S"


def decode_time(timestamp: str) -> cftime.DatetimeJulian:
    return datetime.datetime.strptime(timestamp, TIME_FMT)


def encode_time(time: cftime.DatetimeJulian) -> str:
    return time.strftime(TIME_FMT)


def time_to_list(t: cftime.DatetimeJulian) -> List[int]:
    """Retrieve the 'current_date' in the format required by fv3gfs namelist
    from timestamp string."""
    return [t.year, t.month, t.day, t.hour, t.minute, t.second]