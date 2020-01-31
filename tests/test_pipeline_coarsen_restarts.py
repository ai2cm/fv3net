import pytest
import os

from fv3net.pipelines.coarsen_restarts.pipeline import check_coarsen_incomplete

TEST_DATA_GCS = "gs://vcm-ml-data/fv3net-testing-data/coarsen-timesteps"
SRC_GCS = os.path.join(TEST_DATA_GCS, "C384")
TIMESTEP = "20160801.001500"
GRIDSPEC_GCS = os.path.join(TEST_DATA_GCS, "gridspec-c384")
COMPARE_GCS = os.path.join(TEST_DATA_GCS, "target-C48", TIMESTEP)
DST_GCS = os.path.join(TEST_DATA_GCS, "C48")


@pytest.mark.parametrize(
    "check_gcs_url, expected",
    [
        (f"gs://vcm-ml-data/non/existent/folder/for/timesteps/", True),
        (f"{TEST_DATA_GCS}/incomplete/partial/", True),
        (f"{TEST_DATA_GCS}/target-C48/", False),
        (f"{TEST_DATA_GCS}/incomplete/empty/", True),
    ],
)
def test_coarsen_timestep_folder_completion(check_gcs_url, expected):
    result = check_coarsen_incomplete(COMPARE_GCS, check_gcs_url)
    assert result == expected
