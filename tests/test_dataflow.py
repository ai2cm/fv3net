import pytest

from fv3net.pipelines.examples import simple_fv3net

"""
Runs a simple dataflow test to ensure that job submission with base
fv3net packages works.
"""

PIPELINE_ARGS = [
    "--runner",
    "DataflowRunner",
    "--job-name",
    "simple-dataflow-regression-test" "--project",
    "vcm-ml",
    "--region",
    "us-central1",
    "--temp_location",
    "gs://vcm-ml-data/tmp_dataflow",
    "--num_workers",
    "1",
    "--max_num_workers",
    "1",
    "--disk_size_gb",
    "30",
    "--worker_machine_type",
    "n1-standard-1",
    "--setup_file",
    "./setup.py",
    "--extra_package",
    "external/vcm/dist/vcm-0.1.0.tar.gz",
    "--extra_package",
    "external/vcm/external/mappm/dist/mappm-0.0.0.tar.gz",
]

TEST_GCS_OUT = "gs://vcm-ml-data/fv3net-testing-data/simple-dataflow"


@pytest.mark.regression
def test_simple_dataflow_job():
    list_to_sum = list(range(11))
    simple_fv3net.run(list_to_sum, TEST_GCS_OUT, pipeline_options=PIPELINE_ARGS)
    simple_fv3net.check_run(TEST_GCS_OUT, sum(list_to_sum))


if __name__ == "__main__":
    test_simple_dataflow_job()
