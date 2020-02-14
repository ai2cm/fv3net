import logging
import os
import uuid
import yaml
import fsspec
import re
from multiprocessing import Pool
from typing import List, Mapping

import fv3config

from vcm.convenience import list_timesteps
from vcm.cubedsphere.constants import RESTART_CATEGORIES, TILE_COORDS_FILENAMES
from vcm.cloud.fsspec import get_protocol, get_fs

# TODO: This top level gets adjusted by an update function below, should change
KUBERNETES_CONFIG_DEFAULT = {
    "docker_image": "us.gcr.io/vcm-ml/fv3gfs-python",
    "cpu_count": 6,
    "gcp_secret": "gcp-key",
    "image_pull_policy": "Always",
}
MODEL_CONFIG_DEFAULT = fv3config.get_default_config()


SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = SECONDS_IN_MINUTE * 60
SECONDS_IN_DAY = SECONDS_IN_HOUR * 24

STDOUT_FILENAME = "stdout.log"
VERTICAL_GRID_FILENAME = "fv_core.res.nc"

logger = logging.getLogger("one_step_jobs")


def update_nested_dict(source_dict: dict, update_dict: dict) -> dict:
    for key in update_dict:
        if key in source_dict and isinstance(source_dict[key], Mapping):
            update_nested_dict(source_dict[key], update_dict[key])
        else:
            source_dict[key] = update_dict[key]
    return source_dict


def upload_if_necessary(path: str, remote_url: str) -> str:
    if get_protocol(path) == "file":
        remote_path = os.path.join(remote_url, os.path.basename(path))
        get_fs(remote_url).put(path, remote_path)
        path = remote_path
    return path


# Timestep Handling
def add_timestep_to_url(url: str, timestep: str) -> str:
    return os.path.join(url, timestep)


def _current_date_from_timestep(timestep: str) -> List[int]:
    """Return timestep in the format required by fv3gfs namelist"""
    year = int(timestep[:4])
    month = int(timestep[4:6])
    day = int(timestep[6:8])
    hour = int(timestep[9:11])
    minute = int(timestep[11:13])
    second = int(timestep[13:15])
    return [year, month, day, hour, minute, second]


def timesteps_to_process(
    input_url: str, output_url: str, n_steps: int, overwrite: bool
) -> List[str]:
    """Return list of timesteps left to process. This is all the timesteps in
    input_url minus the successfully completed timesteps in output_url. List is
    also limited to a length of n_steps (which can be None, i.e. no limit)"""
    rundirs_url = os.path.join(output_url)
    to_do = list_timesteps(input_url)
    done = _check_runs_complete(rundirs_url)
    if overwrite:
        _delete_logs_of_done_timesteps(output_url, done)
        done = []
    timestep_list = sorted(list(set(to_do) - set(done)))[:n_steps]
    logger.info(f"Number of input times: {len(to_do)}")
    logger.info(f"Number of completed times: {len(done)}")
    logger.info(f"Number of times to process: {len(timestep_list)}")
    return timestep_list


def _delete_logs_of_done_timesteps(output_url: str, timesteps: List[str]):
    fs = get_fs(output_url)
    for timestep in timesteps:
        rundir_url = os.path.join(output_url, timestep)
        fs.rm(os.path.join(rundir_url, STDOUT_FILENAME))


# Run Completion Checks
def _check_runs_complete(rundir_url: str):
    """Checks for existence of stdout.log and tail of said logfile to see if run
    successfully completed"""
    # TODO: Ideally this would check some sort of model exit code
    timestep_check_args = [
        (curr_timestep, os.path.join(rundir_url, curr_timestep, STDOUT_FILENAME))
        for curr_timestep in list_timesteps(rundir_url)
    ]
    pool = Pool(processes=16)
    complete = set(pool.map(_check_run_complete_unpacker, timestep_check_args))
    pool.close()
    if None in complete:
        complete.remove(None)
    return complete


def _check_run_complete_unpacker(arg: tuple) -> str:
    return _check_run_complete_func(*arg)


def _check_run_complete_func(timestep: str, logfile_path: str) -> str:
    if get_fs(logfile_path).exists(logfile_path) and _check_log_tail(logfile_path):
        return timestep
    else:
        return None


def _check_log_tail(gcs_log_file: str) -> bool:
    # Checking for specific timing headers that output at the end of the run
    output_timing_header = [
        "tmin",
        "tmax",
        "tavg",
        "tstd",
        "tfrac",
        "grain",
        "pemin",
        "pemax\n",
    ]
    output_timing_row_lead = [
        "Total",
        "Initialization",
        "FV",
        "FV",
        "FV",
        "GFS",
        "GFS",
        "GFS",
        "Dynamics",
        "Dynamics",
        "FV3",
        "Main",
        "Termination",
    ]
    with fsspec.open(gcs_log_file, "r") as f:
        log_output = f.readlines()[-15:]
    headers = [word for word in log_output[0].split(" ") if word]
    top_check = _check_header_categories(output_timing_header, headers)
    row_headers = [line.split(" ")[0] for line in log_output[1:-1]]
    row_check = _check_header_categories(output_timing_row_lead, row_headers)
    return top_check and row_check


def _check_header_categories(
    target_categories: List[str], source_categories: List[str]
) -> bool:
    if not source_categories:
        return False
    elif len(source_categories) != len(target_categories):
        return False
    for i, target in enumerate(target_categories):
        if target != source_categories[i]:
            return False
    return True


# Config Handling
def _get_initial_condition_assets(input_url: str, timestep: str) -> List[dict]:
    """Get list of assets representing initial conditions for this timestep to pipeline for now"""
    initial_condition_assets = [
        fv3config.get_asset_dict(
            input_url,
            f"{timestep}.{category}.tile{tile}.nc",
            target_location="INPUT",
            target_name=f"{category}.tile{tile}.nc",
        )
        for category in RESTART_CATEGORIES
        for tile in TILE_COORDS_FILENAMES
    ]
    return initial_condition_assets


def _get_vertical_grid_asset(config_url: str) -> dict:
    """Get asset corresponding to fv_core.res.nc which describes vertical coordinate"""
    return fv3config.get_asset_dict(
        config_url,
        VERTICAL_GRID_FILENAME,
        target_location="INPUT",
        target_name=VERTICAL_GRID_FILENAME,
    )


# TODO: Specific to One-step
def _get_experiment_name(first_tag: str, second_tag: str) -> str:
    """Return unique experiment name with only alphanumeric, hyphen or dot"""
    uuid_short = str(uuid.uuid4())[:8]
    exp_name = f"one-step.{first_tag}.{second_tag}.{uuid_short}"
    return re.sub(r"[^a-zA-Z0-9.\-]", "", exp_name)


def _update_and_upload_config(
    workflow_name: str,
    one_step_config: dict,
    input_url: str,
    config_url: str,
    timestep: str,
    local_vertical_grid_file=None,
) -> dict:
    
    user_model_config = one_step_config["fv3config"]
    user_kubernetes_config = one_step_config["kubernetes"]
    model_config = update_nested_dict(MODEL_CONFIG_DEFAULT, user_model_config)
    kubernetes_config = update_nested_dict(
        KUBERNETES_CONFIG_DEFAULT, user_kubernetes_config
    )

    # Set restart and adjust and upload assets to GCS
    model_config = prepare_and_upload_model_config(
        workflow_name, input_url, config_url, timestep, model_config,
        local_vertical_grid_file=local_vertical_grid_file
    )

    if "runfile" in kubernetes_config:
        runfile_path = kubernetes_config["runfile"]
        kubernetes_config["runfile"] = upload_if_necessary(
            runfile_path, config_url
        )

    return {"fv3config": model_config, "kubernetes": kubernetes_config}
    

def prepare_and_upload_model_config(
    workflow_name: str,
    input_url: str,
    config_url: str,
    timestep: str,
    model_config: dict,
    upload_config_filename="fv3config.yml",
    local_vertical_grid_file=None
) -> dict:
    """Update model and kubernetes configurations for this particular
    timestep and upload necessary files to GCS"""

    model_config = fv3config.enable_restart(model_config)
    model_config["experiment_name"] = _get_experiment_name(workflow_name, timestep)
    model_config["initial_conditions"] = _get_initial_condition_assets(
        input_url, timestep
    )
    model_config["initial_conditions"].append(_get_vertical_grid_asset(config_url))
    model_config["namelist"]["coupler_nml"].update(
        {
            "current_date": _current_date_from_timestep(timestep),
            "force_date_from_namelist": True,
        }
    )

    model_config["diag_table"] = upload_if_necessary(
        model_config["diag_table"], config_url
    )

    if local_vertical_grid_file is not None:
        fs = get_fs(config_url)
        fs.put(local_vertical_grid_file, os.path.join(config_url, VERTICAL_GRID_FILENAME))

    with fsspec.open(os.path.join(config_url, upload_config_filename), "w") as config_file:
        config_file.write(yaml.dump(model_config))

    return model_config


def submit_jobs(
    timestep_list: List[str],
    workflow_name: str,
    one_step_config: dict,
    input_url: str,
    output_url: str,
    config_url: str,
    experiment_label=None,
    local_vertical_grid_file=None,
):
    """Submit one-step job for all timesteps in timestep_list"""
    for timestep in timestep_list:

        curr_input_url = add_timestep_to_url(input_url, timestep)
        curr_output_url = add_timestep_to_url(output_url, timestep)
        curr_config_url = add_timestep_to_url(config_url, timestep)

        one_step_config = _update_and_upload_config(
            workflow_name, one_step_config, curr_input_url, curr_config_url, timestep,
            local_vertical_grid_file=local_vertical_grid_file
        )

        jobname = one_step_config["fv3config"]["experiment_name"]
        one_step_config["kubernetes"]["jobname"] = jobname
        fv3config.run_kubernetes(
            os.path.join(curr_config_url, "fv3config.yml"),
            curr_output_url,
            experiment_label=experiment_label,
            **one_step_config["kubernetes"],
        )
        logger.info(f"Submitted job {jobname}")
