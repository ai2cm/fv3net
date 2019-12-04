import copy
import logging
import os
import subprocess
import uuid
from datetime import timedelta
from pathlib import Path
from typing import List
import gcsfs
import yaml
import fv3config

logger = logging.getLogger("run_jobs")

PARENT_BUCKET = (
    "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/"
)

INPUT_BUCKET = PARENT_BUCKET + "restart/C48/"
OUTPUT_BUCKET = PARENT_BUCKET + "one_step_output/C48/"
CONFIG_BUCKET = PARENT_BUCKET + "one_step_config/C48/"
DOCKER_IMAGE = "us.gcr.io/vcm-ml/fv3gfs-python"
LOCAL_RUNFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runfile.py")
LOCAL_DIAG_TABLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "diag_table"
)

SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = SECONDS_IN_MINUTE * 60
SECONDS_IN_DAY = SECONDS_IN_HOUR * 24


def set_run_duration(config: dict, duration: timedelta) -> dict:
    return_config = copy.deepcopy(config)
    coupler_nml = return_config["namelist"]["coupler_nml"]
    total_seconds = duration.total_seconds()
    if total_seconds % 1 != 0:
        raise ValueError("duration must be an integer number of seconds")
    coupler_nml["months"] = 0
    coupler_nml["hours"] = 0
    coupler_nml["minutes"] = 0
    days = int(total_seconds / SECONDS_IN_DAY)
    coupler_nml["days"] = days
    coupler_nml["seconds"] = int(total_seconds - (days * SECONDS_IN_DAY))
    return return_config


def timestep_from_url(url):
    return str(Path(url).name)


def list_timesteps(bucket: str) -> List[str]:
    """Returns the unique timesteps in a bucket"""
    ls_output = subprocess.check_output(["gsutil", "ls", bucket])
    file_list = ls_output.decode("UTF-8").strip().split()
    timesteps = map(timestep_from_url, file_list)
    return set(timesteps)


def timesteps_to_process() -> List[str]:
    to_do = list_timesteps(INPUT_BUCKET)
    done = list_timesteps(OUTPUT_BUCKET)
    logger.info(f"Number of input times: {len(to_do)}")
    logger.info(f"Number of completed times: {len(done)}")
    logger.info(f"Number of times to process: {len(to_do) - len(done)}")
    return sorted(list(set(to_do) - set(done)))


def submit_jobs(timestep_list):
    fs = gcsfs.GCSFileSystem(project="VCM-ML")
    for timestep in timestep_list[:1]:
        job_name = f"one-step.{timestep}.{uuid.uuid4()}"
        config = fv3config.get_default_config()
        config = fv3config.enable_restart(config)
        config["experiment_name"] = job_name
        config["initial_conditions"] = os.path.join(
            INPUT_BUCKET, timestep, "rundir/INPUT"
        )
        config = set_run_duration(config, timedelta(minutes=15))
        config["namelist"]["coupler_nml"].update({"dt_atmos": 60, "dt_ocean": 60})
        config["namelist"]["fv_core_nml"].update(
            {"external_eta": True, "npz": 79, "k_split": 1, "n_split": 1}
        )
        config_location = os.path.join(CONFIG_BUCKET, timestep, "fv3config.yml")
        with fs.open(config_location, "w") as config_file:
            config_file.write(yaml.dump(config))
        runfile_location = os.path.join(CONFIG_BUCKET, timestep, "runfile.py")
        fs.put(LOCAL_RUNFILE, runfile_location)
        diag_table_location = os.path.join(CONFIG_BUCKET, timestep, "diag_table")
        fs.put(LOCAL_DIAG_TABLE, diag_table_location)
        config["diag_table"] = diag_table_location
        outdir = os.path.join(OUTPUT_BUCKET, timestep)
        fv3config.run_kubernetes(
            config_location,
            outdir,
            DOCKER_IMAGE,  # runfile=runfile_location,
            jobname=job_name,
            namespace="default",
            memory_gb=3.6,
            cpu_count=1,
            gcp_secret="gcp-key",
            image_pull_policy="Always",
        )
        logger.info(f"Submitted job for timestep {timestep}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    timestep_list = timesteps_to_process()
    submit_jobs(timestep_list)
