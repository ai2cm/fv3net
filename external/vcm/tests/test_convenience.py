import pytest
from datetime import datetime
import xarray as xr
import cftime

import vcm
from vcm.cubedsphere.constants import TIME_FMT
from vcm.convenience import (
    parse_timestep_str_from_path,
    parse_datetime_from_str,
)


def test_extract_timestep_from_path():

    timestep = "20160801.001500"
    good_path = f"gs://path/to/timestep/{timestep}/"
    assert parse_timestep_str_from_path(good_path) == timestep


def test_extract_timestep_from_path_with_no_timestep_in_path():

    with pytest.raises(ValueError):
        bad_path = "gs://path/to/not/a/timestep/"
        parse_timestep_str_from_path(bad_path)


def test_datetime_from_string():

    current_time = datetime.now()
    time_str = current_time.strftime(TIME_FMT)
    parsed_datetime = parse_datetime_from_str(time_str)

    assert parsed_datetime.year == current_time.year
    assert parsed_datetime.month == current_time.month
    assert parsed_datetime.day == current_time.day
    assert parsed_datetime.hour == current_time.hour
    assert parsed_datetime.minute == current_time.minute
    assert parsed_datetime.second == current_time.second


def test_convert_timestamps():
    arr = xr.DataArray(["20190101.000000", "20160604.011500"], attrs={"foo": "bar"})
    out = vcm.convert_timestamps(arr)
    assert isinstance(out[0].item(), cftime.DatetimeJulian)
    assert out.attrs == arr.attrs
