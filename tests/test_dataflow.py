from fv3net.pipelines.examples import simple_fv3net

PIPELINE_ARGS = [
    "--runner", "DataflowRunner",
    "--project", "vcm-ml",
    "--region", "us-central1",
    "--temp_location", "gs://vcm-ml-data/tmp_dataflow",
    "--num_workers", "1",
    "--max_num_workers", "1",
    "--disk_size_gb", "30",
    "--worker_machine_type", "n1-standard-1",
    "--setup_file", "./setup.py",
    "--extra_package", "external/vcm/dist/vcm-0.1.0.tar.gz",
    "--extra_package", "external/vcm/external/mappm/dist/mappm-0.0.0.tar.gz",
]

TEST_GCS_OUT = "gs://vcm-ml-data/fv3net-testing-data/simple-dataflow"


def test_simple_dataflow():
    sum_list = list(range(10))
    simple_fv3net.run(sum_list, TEST_GCS_OUT, pipeline_options=PIPELINE_ARGS)


if __name__ == "__main__":
    test_simple_dataflow()
