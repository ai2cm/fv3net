import logging
import os
import fsspec
import uuid
import yaml
import re
from copy import deepcopy
from multiprocessing import Pool
from typing import List, Tuple

import fv3config
from . import utils
from ..common import list_timesteps, subsample_timesteps_at_interval
from vcm.cloud.fsspec import get_fs

STDOUT_FILENAME = "stdout.log"
VERTICAL_GRID_FILENAME = "fv_core.res.nc"

SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = SECONDS_IN_MINUTE * 60
SECONDS_IN_DAY = SECONDS_IN_HOUR * 24

KUBERNETES_CONFIG_DEFAULT = {
    "docker_image": "us.gcr.io/vcm-ml/fv3gfs-python",
    "cpu_count": 6,
    "gcp_secret": "gcp-key",
    "image_pull_policy": "Always",
}

logger = logging.getLogger(__name__)


def create_zarr_store()


def timesteps_to_process(
    input_url: str,
    output_url: str,
    n_steps: int,
    overwrite: bool,
    subsample_frequency: int = None,
) -> List[str]:
    """
    Return list of timesteps left to process. This is all the timesteps in
    input_url at the subsampling frequency (if specified) minus the
    successfully completed timesteps in output_url. List is
    also limited to a length of n_steps (which can be None, i.e. no limit)
    """
    rundirs_url = os.path.join(output_url)
    to_do = list_timesteps(input_url)
    if subsample_frequency is not None:
        to_do = subsample_timesteps_at_interval(to_do, subsample_frequency)
    done = check_runs_complete(rundirs_url)
    if overwrite:
        _delete_logs_of_done_timesteps(output_url, done)
        done = []
    timestep_list = sorted(list(set(to_do) - set(done)))[:n_steps]

    logger.info(f"Number of input times: {len(to_do)}")
    logger.info(f"Number of completed times: {min(len(done), len(to_do))}")
    logger.info(f"Number of times to process: {len(timestep_list)}")
    return timestep_list


def _current_date_from_timestep(timestep: str) -> List[int]:
    """Return timestep in the format required by fv3gfs namelist"""
    year = int(timestep[:4])
    month = int(timestep[4:6])
    day = int(timestep[6:8])
    hour = int(timestep[9:11])
    minute = int(timestep[11:13])
    second = int(timestep[13:15])
    return [year, month, day, hour, minute, second]


def _delete_logs_of_done_timesteps(output_url: str, timesteps: List[str]):
    fs = get_fs(output_url)
    for timestep in timesteps:
        rundir_url = os.path.join(output_url, timestep)
        fs.rm(os.path.join(rundir_url, STDOUT_FILENAME))


# Run Completion Checks
def check_runs_complete(rundir_url: str):
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
    """
    Check the tail of an FV3GFS stdout log for output we expect upon successful
    completion.
    """

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


# Configuration Handling


def _get_initial_condition_assets(input_url: str, timestep: str) -> List[dict]:
    """
    Get list of assets representing initial conditions for this timestep to pipeline.
    """
    initial_condition_assets = utils.update_tiled_asset_names(
        source_url=input_url,
        source_filename="{timestep}.{category}.tile{tile}.nc",
        target_url="INPUT",
        target_filename="{category}.tile{tile}.nc",
        timestep=timestep,
    )

    return initial_condition_assets


def _get_vertical_grid_asset(config_url: str) -> dict:
    """Get asset corresponding to fv_core.res.nc which describes vertical coordinate"""
    return fv3config.get_asset_dict(
        config_url,
        VERTICAL_GRID_FILENAME,
        target_location="INPUT",
        target_name=VERTICAL_GRID_FILENAME,
    )


def _get_experiment_name(first_tag: str, second_tag: str) -> str:
    """Return unique experiment name with only alphanumeric, hyphen or dot"""
    uuid_short = str(uuid.uuid4())[:8]
    exp_name = f"one-step.{first_tag}.{second_tag}.{uuid_short}"
    return re.sub(r"[^a-zA-Z0-9.\-]", "", exp_name)


def _update_config(
    workflow_name: str,
    base_config_version: str,
    user_model_config: dict,
    user_kubernetes_config: dict,
    input_url: str,
    config_url: str,
    timestep: str,
) -> Tuple[dict]:
    """
    Update kubernetes and fv3 configurations with user inputs
    to prepare for fv3gfs one-step runs.
    """
    base_model_config = utils.get_base_fv3config(base_config_version)
    model_config = utils.update_nested_dict(base_model_config, user_model_config)
    kubernetes_config = utils.update_nested_dict(
        deepcopy(KUBERNETES_CONFIG_DEFAULT), user_kubernetes_config
    )

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

    return model_config, kubernetes_config


def _upload_config_files(
    model_config: dict,
    kubernetes_config: dict,
    config_url: str,
    local_vertical_grid_file=None,
    upload_config_filename="fv3config.yml",
) -> Tuple[dict]:
    """
    Upload any files to remote paths necessary for fv3config and the
    fv3gfs one-step runs.
    """

    if "runfile" in kubernetes_config:
        runfile_path = kubernetes_config["runfile"]
        kubernetes_config["runfile"] = utils.transfer_local_to_remote(
            runfile_path, config_url
        )

    model_config["diag_table"] = utils.transfer_local_to_remote(
        model_config["diag_table"], config_url
    )

    if local_vertical_grid_file is not None:
        fs = get_fs(config_url)
        vfile_path = os.path.join(config_url, VERTICAL_GRID_FILENAME)
        fs.put(local_vertical_grid_file, vfile_path)

    config_path = os.path.join(config_url, upload_config_filename)
    with fsspec.open(config_path, "w") as config_file:
        config_file.write(yaml.dump(model_config))

    return model_config, kubernetes_config


def prepare_and_upload_config(
    workflow_name: str,
    input_url: str,
    config_url: str,
    timestep: str,
    one_step_config: dict,
    base_config_version: str,
    **kwargs,
) -> Tuple[dict]:
    """Update model and kubernetes configurations for this particular
    timestep and upload necessary files to GCS"""

    user_model_config = one_step_config["fv3config"]
    user_kubernetes_config = one_step_config["kubernetes"]

    model_config, kube_config = _update_config(
        workflow_name,
        base_config_version,
        user_model_config,
        user_kubernetes_config,
        input_url,
        config_url,
        timestep,
    )
    model_config, kube_config = _upload_config_files(
        model_config, kube_config, config_url, **kwargs
    )

    return model_config, kube_config


def submit_jobs(
    timestep_list: List[str],
    workflow_name: str,
    one_step_config: dict,
    input_url: str,
    output_url: str,
    config_url: str,
    base_config_version: str,
    job_labels=None,
    local_vertical_grid_file=None,
) -> None:
    """Submit one-step job for all timesteps in timestep_list"""
    for timestep in timestep_list:

        curr_input_url = os.path.join(input_url, timestep)
        curr_output_url = os.path.join(output_url, timestep)
        curr_config_url = os.path.join(config_url, timestep)

        model_config, kube_config = prepare_and_upload_config(
            workflow_name,
            curr_input_url,
            curr_config_url,
            timestep,
            one_step_config,
            base_config_version,
            local_vertical_grid_file=local_vertical_grid_file,
        )

        jobname = model_config["experiment_name"]
        kube_config["jobname"] = jobname
        fv3config.run_kubernetes(
            os.path.join(curr_config_url, "fv3config.yml"),
            curr_output_url,
            job_labels=job_labels,
            **kube_config,
        )
        logger.info(f"Submitted job {jobname}")
