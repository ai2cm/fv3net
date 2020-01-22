import copy
import logging
import os
import uuid
from datetime import datetime, timedelta
import gcsfs
import yaml
import fv3config

logger = logging.getLogger("run_jobs")

START_DATE = datetime(2016, 1, 1, 0, 0, 0)
RUN_DURATION = timedelta(days=5)

RUN_NAME = f"free-run-2016.{uuid.uuid4()}"
BUCKET = "gs://vcm-ml-data/"
IC_BUCKET = BUCKET + "2019-12-03-C48-20160101.00Z_IC"
OUTPUT_BUCKET = BUCKET + f"2019-12-12-baseline-FV3GFS-runs/free/C48/{RUN_NAME}/output"
CONFIG_BUCKET = BUCKET + f"2019-12-12-baseline-FV3GFS-runs/free/C48/{RUN_NAME}/config"
DOCKER_IMAGE = "us.gcr.io/vcm-ml/fv3gfs-python"
LOCAL_RUNFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runfile.py")
LOCAL_DIAG_TABLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "diag_table_for_long_run"
)


def date_to_list(date: datetime) -> list:
    return [date.year, date.month, date.day, date.hour, date.minute, date.second]


def get_config(start_date: datetime, run_duration: timedelta) -> dict:
    config = fv3config.get_default_config()
    config["experiment_name"] = RUN_NAME
    config["diag_table"] = os.path.join(CONFIG_BUCKET, "diag_table")
    config["initial_conditions"] = IC_BUCKET
    config = fv3config.set_run_duration(config, RUN_DURATION)
    # make physics diagnostic output hourly
    config["namelist"]["atmos_model_nml"]["fhout"] = 1.0
    config["namelist"]["coupler_nml"].update(
        {"current_date": date_to_list(START_DATE), "dt_atmos": 900, "dt_ocean": 900}
    )
    config["namelist"]["fv_core_nml"]["layout"] = [2, 2]
    return config


def submit_job() -> None:
    fs = gcsfs.GCSFileSystem(project="VCM-ML")
    config = get_config(START_DATE, RUN_DURATION)
    job_name = config["experiment_name"]
    config_location = os.path.join(CONFIG_BUCKET, "fv3config.yml")
    with fs.open(config_location, "w") as config_file:
        config_file.write(yaml.dump(config))
    runfile_location = os.path.join(CONFIG_BUCKET, "runfile.py")
    fs.put(LOCAL_RUNFILE, runfile_location)
    fs.put(LOCAL_DIAG_TABLE, os.path.join(CONFIG_BUCKET, "diag_table"))
    fv3config.run_kubernetes(
        config_location,
        OUTPUT_BUCKET,
        DOCKER_IMAGE,
        runfile=runfile_location,
        jobname=job_name,
        namespace="default",
        memory_gb=14,
        cpu_count=24,
        gcp_secret="gcp-key",
        image_pull_policy="Always",
    )
    logger.info(f"Submitted {job_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    submit_job()
