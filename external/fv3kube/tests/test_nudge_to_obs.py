from datetime import timedelta
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
    assert "url" in output["gfs_analysis_data"]
    assert "filename_pattern" in output["gfs_analysis_data"]


def test_enable_nudge_to_observations_adds_nudging_asset():
    current_date = [2016, 1, 2, 1, 0, 0]
    duration = timedelta(days=1)
    url = "/path/to/nudging/files"
    pattern = "%Y%m%d_%HZ_T85LR.nc"

    output = fv3kube.enable_nudge_to_observations(
        duration, current_date, nudge_url=url, nudge_filename_pattern=pattern
    )

    patch_files = output["patch_files"]
    assert any(asset["target_name"] == "20160102_00Z_T85LR.nc" for asset in patch_files)
