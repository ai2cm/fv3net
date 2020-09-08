import pytest
from datetime import datetime, timedelta

from fv3kube import nudge_to_obs
import fv3kube


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
    assert nudge_to_obs._most_recent_nudge_time(start_time) == expected


@pytest.mark.parametrize(
    "duration, current_date, expected_length, expected_first_datetime, expected_last_datetime",
    [
        (
            timedelta(days=1),
            [2016, 1, 1, 0, 0, 0],
            4 + 1,
            datetime(2016, 1, 1),
            datetime(2016, 1, 2),
        ),
        (
            timedelta(days=1, hours=5),
            [2016, 1, 1, 0, 0, 0],
            4 + 1 + 1,
            datetime(2016, 1, 1),
            datetime(2016, 1, 2, 6),
        ),
        (
            timedelta(days=1, hours=7),
            [2016, 1, 1, 0, 0, 0],
            4 + 2 + 1,
            datetime(2016, 1, 1),
            datetime(2016, 1, 2, 12),
        ),
        (
            timedelta(days=1),
            [2016, 1, 2, 1, 0, 0],
            4 + 2,
            datetime(2016, 1, 2),
            datetime(2016, 1, 3, 6),
        ),
    ],
)
def test__get_nudge_time_list(
    duration,
    current_date,
    expected_length,
    expected_first_datetime,
    expected_last_datetime,
):
    nudge_file_list = nudge_to_obs._get_nudge_time_list(duration, current_date)
    assert len(nudge_file_list) == expected_length
    assert nudge_file_list[0] == expected_first_datetime
    assert nudge_file_list[-1] == expected_last_datetime


def test_enable_nudge_to_observations_namelist_options():
    """Test that the nudging namelist options are added

    This test doesn't comprehensively cover all the options. It ensures that
    some options are correctly set, which can detect implementation problems.
    """
    current_date = [2016, 1, 2, 1, 0, 0]
    duration = timedelta(days=10)
    output = fv3kube.enable_nudge_to_observations(duration, current_date)
    assert output["namelist"]["fv_core_nml"]["nudge"]


def test_enable_nudge_to_observations_adds_filelist_asset():
    current_date = [2016, 1, 2, 1, 0, 0]
    duration = timedelta(days=10)

    file_list_path = "asdf"

    output = fv3kube.enable_nudge_to_observations(
        duration, current_date, file_list_path=file_list_path
    )

    # check patch files
    patch_files = output["patch_files"]
    # search for filelist asset
    for asset in patch_files:
        if asset["target_name"] == file_list_path:
            found = asset

    assert "bytes" in found
