import pytest
from datetime import datetime

from fv3net.pipelines.kube_jobs import nudge_to_obs

HOURS_IN_DAY = 24
NUDGE_FILES_PER_DAY = HOURS_IN_DAY / nudge_to_obs.NUDGE_INTERVAL


@pytest.mark.parametrize(
    "current_date, expected_datetime",
    [
        ([2016, 1, 1, 0, 0, 0], datetime(2016, 1, 1)),
        ([2016, 1, 1, 1, 0, 0], datetime(2016, 1, 1, 1)),
        ([2016, 1, 1, 7, 2, 0], datetime(2016, 1, 1, 7, 2)),
        ([2016, 1, 1, 12, 3, 10], datetime(2016, 1, 1, 12, 3, 10)),
    ],
)
def test__datetime_from_current_date(current_date, expected_datetime):
    assert nudge_to_obs._datetime_from_current_date(current_date) == expected_datetime


@pytest.mark.parametrize(
    "start_time, expected",
    [
        (datetime(2016, 1, 1), datetime(2016, 1, 1, 0)),
        (datetime(2016, 1, 1, 1), datetime(2016, 1, 1, 0)),
        (datetime(2016, 1, 1, 7), datetime(2016, 1, 1, 6)),
        (datetime(2016, 1, 1, 12), datetime(2016, 1, 1, 12)),
        (datetime(2016, 1, 2, 18, 1), datetime(2016, 1, 2, 18)),
    ],
)
def test__get_first_nudge_file_time(start_time, expected):
    assert nudge_to_obs._get_first_nudge_time(start_time) == expected


@pytest.mark.parametrize(
    "coupler_nml, expected_length, expected_first_datetime, expected_last_datetime",
    [
        (
            {"current_date": [2016, 1, 1, 0, 0, 0], "days": 1},
            4 + 1,
            datetime(2016, 1, 1),
            datetime(2016, 1, 2),
        ),
        (
            {"current_date": [2016, 1, 1, 0, 0, 0], "days": 1, "hours": 5},
            4 + 1 + 1,
            datetime(2016, 1, 1),
            datetime(2016, 1, 2, 6),
        ),
        (
            {"current_date": [2016, 1, 1, 0, 0, 0], "days": 1, "hours": 7},
            4 + 2 + 1,
            datetime(2016, 1, 1),
            datetime(2016, 1, 2, 12),
        ),
        (
            {"current_date": [2016, 1, 2, 1, 0, 0], "days": 1},
            4 + 2,
            datetime(2016, 1, 2),
            datetime(2016, 1, 3, 6),
        ),
    ],
)
def test__get_nudge_time_list(
    coupler_nml, expected_length, expected_first_datetime, expected_last_datetime
):
    config = {"namelist": {"coupler_nml": coupler_nml}}
    nudge_file_list = nudge_to_obs._get_nudge_time_list(config)
    assert len(nudge_file_list) == expected_length
    assert nudge_file_list[0] == expected_first_datetime
    assert nudge_file_list[-1] == expected_last_datetime
