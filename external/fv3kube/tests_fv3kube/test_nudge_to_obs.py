import pytest
from datetime import datetime, timedelta

from fv3kube import nudge_to_obs
import fv3kube


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
