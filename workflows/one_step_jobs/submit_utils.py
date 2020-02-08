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


SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = SECONDS_IN_MINUTE * 60
SECONDS_IN_DAY = SECONDS_IN_HOUR * 24

logger = logging.getLogger("one_step_jobs")


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


def timestep_from_url(url: str) -> str:
    return str(Path(url).name)


def list_timesteps(bucket: str) -> List[str]:
    """Returns the unique timesteps in a bucket"""
    ls_output = subprocess.check_output(["gsutil", "ls", bucket])
    file_list = ls_output.decode("UTF-8").strip().split()
    timesteps = map(timestep_from_url, file_list)
    return set(timesteps)


def timesteps_to_process(input_bucket: str, output_bucket: str) -> List[str]:
    to_do = list_timesteps(input_bucket)
    done = list_timesteps(output_bucket)
    logger.info(f"Number of input times: {len(to_do)}")
    logger.info(f"Number of completed times: {len(done)}")
    logger.info(f"Number of times to process: {len(to_do) - len(done)}")
    return sorted(list(set(to_do) - set(done)))


def get_config(timestep: str, input_bucket: str):
    config = fv3config.get_default_config()
    config = fv3config.enable_restart(config)
    config["experiment_name"] = f"one-step.{timestep}.{uuid.uuid4()}"
    config["initial_conditions"] = os.path.join(input_bucket, timestep, "rundir/INPUT")
    config = set_run_duration(config, timedelta(minutes=15))
    config["namelist"]["coupler_nml"].update({"dt_atmos": 60, "dt_ocean": 60})
    config["namelist"]["fv_core_nml"].update(
        {"external_eta": True, "npz": 79, "k_split": 1, "n_split": 1}
    )
    return config


def submit_jobs(
    timestep_list: List[str],
    config_bucket: str,
    local_runfile: str,
    local_diag_table: str,
    docker_image: str,
    output_bucket: str,
    experiment_label: str = None,
) -> None:
    fs = gcsfs.GCSFileSystem()
    for timestep in timestep_list:
        config = get_config(timestep)
        job_name = config["experiment_name"]
        runfile_location = os.path.join(config_bucket, timestep, "runfile.py")
        fs.put(local_runfile, runfile_location)
        diag_table_location = os.path.join(config_bucket, timestep, "diag_table")
        fs.put(local_runfile, diag_table_location)
        config["diag_table"] = diag_table_location
        config_location = os.path.join(config_bucket, timestep, "fv3config.yml")
        with fs.open(config_location, "w") as config_file:
            config_file.write(yaml.dump(config))
        outdir = os.path.join(output_bucket, timestep)
        fv3config.run_kubernetes(
            config_location,
            outdir,
            docker_image,
            runfile=runfile_location,
            jobname=job_name,
            namespace="default",
            memory_gb=3.6,
            cpu_count=1,
            gcp_secret="gcp-key",
            image_pull_policy="Always",
            experiment_label=experiment_label,
        )
        logger.info(f"Submitted job for timestep {timestep}")
