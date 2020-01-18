import pytest
import subprocess
import tempfile
import os
from vcm.cloud import gsutil

import misc.utils as mutil
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
    subprocess.check_call(['gsutil', '-m', 'rm', '-r', DST_GCS])
    

# test coarsen, check it that creates C48 directory with all files checksummed
@pytest.mark.regression
def test_coarsen_timestep_single_coarsen_operation(temporary_gcs_dst):

    coarsen_timestep(
        SRC_GCS + TIMESTEP, temporary_gcs_dst, 384 // 48, GRIDSPEC_GCS
    )

    assert gsutil.exists(temporary_gcs_dst)

    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as chkdir:
        
        # Get test coarsened files
        gsutil.copy_directory_contents(temporary_gcs_dst, tmpdir)

        target_md5_file = os.path.join(chkdir, 'checksum')
        gsutil.copy(os.path.join(TEST_DATA_GCS, 'checksum'),
                    target_md5_file)
        
        with open(target_md5_file, 'r') as f:
            target_md5 = f.readlines()[0]
        coarsened_md5 = mutil.calc_directory_md5(tmpdir)

        assert target_md5 == coarsened_md5


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
