from getpass import getuser
from fv3net.pipelines.common import get_alphanumeric_unique_tag

COARSEN_RESTARTS_DATAFLOW_ARGS = {
    "--job_name": (
        f"coarsen-restarts-{getuser().lower()}-{get_alphanumeric_unique_tag(7)}"
    ),
    "--project": "vcm-ml",
    "--region": "us-central1",
    "--temp_location": "gs://vcm-ml-data/tmp_dataflow",
    "--num_workers": 3,
    "--max_num_workers": 50,
    "--disk_size_gb": 50,
    "--worker_machine_type": "n1-highmem-4",
    "--setup_file": "./setup.py",
    "--extra_package": ["external/vcm/external/mappm/dist/mappm-0.0.0.tar.gz"],
}

ONE_STEP_DIAGS_DATAFLOW_ARGS = COARSEN_RESTARTS_DATAFLOW_ARGS.copy()
ONE_STEP_DIAGS_DATAFLOW_ARGS.update(
    {
        "--job_name": (
            f"one-step-diags-{getuser().lower()}-" f"{get_alphanumeric_unique_tag(7)}"
        ),
        "--num_workers": 4,
        "--max_num_workers": 30,
        "--disk_size_gb": 250,
        "--worker_machine_type": "n1-highmem-16",
        "--extra_package": [
            "external/report/dist/report-0.1.0.tar.gz",
        ],
    }
)

CREATE_TRAINING_DATAFLOW_ARGS = COARSEN_RESTARTS_DATAFLOW_ARGS.copy()
CREATE_TRAINING_DATAFLOW_ARGS.update(
    {
        "--job_name": (
            f"create-training-data-{getuser().lower()}-"
            f"{get_alphanumeric_unique_tag(7)}"
        ),
        "--num_workers": 4,
        "--max_num_workers": 30,
        "--disk_size_gb": 30,
        "--worker_machine_type": "n1-standard-1",
    }
)
