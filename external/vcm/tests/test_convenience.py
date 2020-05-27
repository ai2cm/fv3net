import pytest
from datetime import datetime, timedelta
import xarray as xr
import cftime
import numpy as np

import vcm
from vcm.cubedsphere.constants import TIME_FMT
from vcm.convenience import (
    cast_to_datetime,
    parse_current_date_from_str,
    parse_timestep_str_from_path,
    round_time,
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


def test_current_date_from_string():
    timestamp = "20160801.001500"
    expected_current_date = [2016, 8, 1, 0, 15, 0]
    assert expected_current_date == parse_current_date_from_str(timestamp)


def test_convert_timestamps():
    arr = xr.DataArray(["20190101.000000", "20160604.011500"], attrs={"foo": "bar"})
    out = vcm.convert_timestamps(arr)
    assert isinstance(out[0].item(), cftime.DatetimeJulian)
    assert out.attrs == arr.attrs


@pytest.mark.parametrize(
    "input_time, expected",
    [
        (datetime(2016, 1, 1, 1, 1, 1, 1), datetime(2016, 1, 1, 1, 1, 1, 1)),
        (
            cftime.DatetimeJulian(2016, 1, 1, 1, 1, 1, 1),
            datetime(2016, 1, 1, 1, 1, 1, 1),
        ),
        (cftime.DatetimeJulian(2016, 1, 1), datetime(2016, 1, 1)),
        (
            np.datetime64(datetime(2016, 1, 1, 1, 1, 1, 1)),
            datetime(2016, 1, 1, 1, 1, 1, 1),
        ),
    ],
)
def test__cast_to_datetime(input_time, expected):
    casted_input_time = cast_to_datetime(input_time)
    assert casted_input_time == expected
    assert isinstance(casted_input_time, datetime)


def _example_with_second(**kwargs):
    default = dict(year=2016, month=8, day=5, hour=23, minute=7, second=0)
    default.update(kwargs)
    return cftime.DatetimeJulian(**default)


second = timedelta(seconds=1)
minute = timedelta(minutes=1)


@pytest.mark.parametrize(
    "input_,expected,tol",
    [
        (
            _example_with_second(second=29, microsecond=986267),
            _example_with_second(second=30),
            second,
        ),
        (
            _example_with_second(second=59, minute=7, microsecond=986267),
            _example_with_second(second=0, minute=8),
            second,
        ),
        # cftime arithmetic is not associative so need to check this case
        (
            _example_with_second(second=59, minute=7, microsecond=986368),
            _example_with_second(second=59, minute=7, microsecond=986368),
            timedelta(microseconds=1),
        ),
        (
            _example_with_second(minute=7, microsecond=186267),
            _example_with_second(minute=7),
            second,
        ),
        (_example_with_second(minute=7), _example_with_second(minute=7), second,),
        (
            _example_with_second(minute=10),
            _example_with_second(minute=15),
            15 * minute,
        ),
    ],
)
def test_round_time(input_, expected, tol):
    assert round_time(input_, to=tol) == expected
