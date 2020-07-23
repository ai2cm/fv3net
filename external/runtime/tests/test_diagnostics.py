import pytest
from runtime import diagnostics
from datetime import datetime


def test_SelectedTimes():
    times = diagnostics.SelectedTimes({"times": ["20160801.000000"]})
    time = datetime(year=2016, month=8, day=1, hour=0, minute=0, second=0)
    assert time in times


def test_SelectedTimes_not_in_list():
    times = diagnostics.SelectedTimes({"times": ["20160801.000000"]})
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
    times = diagnostics.RegularTimes({"frequency": 900})

    time = datetime(year=2016, month=8, day=1, hour=0, minute=15)
    expected = True
    assert (time in times) == expected


def test_RegularTimes_frequency_over_day_raises_error():
    with pytest.raises(ValueError):
        diagnostics.RegularTimes({"frequency": 86401})
