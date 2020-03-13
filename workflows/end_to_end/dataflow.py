from getpass import getuser

COARSEN_RESTARTS_DATAFLOW_ARGS = [
    ("--job_name", f"coarsen-restarts-{getuser().lower()}"),
    ("--project", "vcm-ml"),
    ("--region", "us-central1"),
    ("--temp_location", "gs://vcm-ml-data/tmp_dataflow"),
    ("--num_workers", 3),
    ("--max_num_workers", 50),
    ("--disk_size_gb", 50),
    ("--worker_machine_type", "n1-highmem-4"),
    ("--setup_file", "./setup.py"),
    ("--extra_package", "external/vcm/dist/vcm-0.1.0.tar.gz"),
    ("--extra_package", "external/vcm/external/mappm/dist/mappm-0.0.0.tar.gz")
]

CREATE_TRAINING_DATAFLOW_ARGS = COARSEN_RESTARTS_DATAFLOW_ARGS.copy()
CREATE_TRAINING_DATAFLOW_ARGS[4:8] = [
    ("--num_workers", 4),
    ("--max_num_workers", 30),
    ("--disk_size_gb", 30),
    ("--worker_machine_type", "n1-standard-1")  
]
del CREATE_TRAINING_DATAFLOW_ARGS[-1]

