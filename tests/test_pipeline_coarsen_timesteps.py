import pytest
import subprocess
import tempfile
import os
import glob
import xarray as xr
from pathlib import Path

from vcm.cloud import gsutil

from fv3net.pipelines.coarsen_timesteps.pipeline import check_timestep_url_incomplete, coarsen_timestep

TEST_DATA_GCS = "gs://vcm-ml-data/fv3net-testing-data/coarsen-timesteps"
SRC_GCS = os.path.join(TEST_DATA_GCS, "C384")
TIMESTEP = "20160801.001500"
GRIDSPEC_GCS = os.path.join(TEST_DATA_GCS, "gridspec-c384")
COMPARE_GCS = os.path.join(TEST_DATA_GCS, "target-C48", TIMESTEP)
DST_GCS = os.path.join(TEST_DATA_GCS, "C48")

@pytest.fixture
def temporary_gcs_dst():

    yield DST_GCS

    # Remove destination directory
    # subprocess.check_call(['gsutil', '-m', 'rm', '-r', DST_GCS])
    

# test coarsen, check it that creates C48 directory with all files checksummed
@pytest.mark.regression
def test_coarsen_timestep_single_coarsen_operation(temporary_gcs_dst):

    # Coarsens timestep and outputs in temporary_gs_dst/TIMESTEP
    coarsen_timestep(
        os.path.join(SRC_GCS, TIMESTEP), temporary_gcs_dst, 384 // 48, GRIDSPEC_GCS
    )

    temporary_gcs_dst_timestep = os.path.join(temporary_gcs_dst, TIMESTEP)
    assert gsutil.exists(temporary_gcs_dst_timestep)

    with tempfile.TemporaryDirectory() as tmpdir:
        
        test_dir = os.path.join(tmpdir, 'test', TIMESTEP)
        os.makedirs(test_dir, exist_ok=True)
        target_dir = os.path.join(tmpdir, 'target', TIMESTEP)
        os.makedirs(target_dir, exist_ok=True)

        gsutil.copy_directory_contents(temporary_gcs_dst_timestep, test_dir)
        gsutil.copy_directory_contents(COMPARE_GCS, target_dir)

        target_files = glob.glob(os.path.join(target_dir, '*'))

        for target_filepath in target_files:
            target_filename = Path(target_filepath).name
            test_filepath = os.path.join(test_dir, target_filename)

            assert os.path.exists(test_filepath)

            target_ds = xr.open_dataset(target_filepath)
            test_ds = xr.open_dataset(test_filepath)

            # DZ and phis have shown relative differences of at least
            # 1e-3 and 1e-2 respectively. Potentially from hydrostatic
            # adjustment as suggested by OliWM. Can't use checksum so
            # allclose for now.
            xr.testing.assert_allclose(target_ds, test_ds, atol=0, rtol=1e-2)


@pytest.mark.parametrize(
    "check_gcs_url, expected",
    [
        (f"gs://vcm-ml-data/non/existent/folder/for/timesteps/", True),
        (f"{TEST_DATA_GCS}/incomplete/partial/", True),
        (f"{TEST_DATA_GCS}/target-C48/", False),
        (f"{TEST_DATA_GCS}/incomplete/empty/", True),
    ]

)
def test_coarsen_timestep_folder_completion(check_gcs_url, expected):
    result = check_timestep_url_incomplete(COMPARE_GCS, check_gcs_url)
    assert result == expected
