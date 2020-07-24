import pytest
from runtime import diagnostics
from datetime import datetime


@pytest.mark.parametrize(
    "time_stamp",
    [
        pytest.param("20160801.000000", id="full time stamp"),
        pytest.param("20160801.00000", id="time stamp with one less 0"),
    ],
)
def test_SelectedTimes(time_stamp):
    times = diagnostics.SelectedTimes([time_stamp])
    time = datetime(year=2016, month=8, day=1, hour=0, minute=0, second=0)
    assert time in times


def test_SelectedTimes_not_in_list():
    times = diagnostics.SelectedTimes(["20160801.000000"])
    time = datetime(year=2016, month=8, day=1, hour=0, minute=0, second=1)
    assert time not in times


@pytest.mark.parametrize(
    "frequency, time, expected",
    [
        (900, datetime(year=2016, month=8, day=1, hour=0, minute=15), True),
        (900, datetime(year=2016, month=8, day=1, hour=0, minute=16), False),
        (900, datetime(year=2016, month=8, day=1, hour=12, minute=45), True),
        (86400, datetime(year=2016, month=8, day=1, hour=0, minute=0), True),
        (86400, datetime(year=2016, month=8, day=2, hour=0, minute=0), True),
    ],
)
def test_RegularTimes(frequency, time, expected):
    times = diagnostics.RegularTimes(frequency)
    assert (time in times) == expected


def test_RegularTimes_frequency_over_day_raises_error():
    with pytest.raises(ValueError):
        diagnostics.RegularTimes(86401)
