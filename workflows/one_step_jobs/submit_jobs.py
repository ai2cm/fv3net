import argparse
from collections.abc import Mapping
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List
from multiprocessing import Pool
import fsspec
import yaml
import fv3config

logger = logging.getLogger("run_jobs")

KUBERNETES_CONFIG_DEFAULT = {
    "docker_image": "us.gcr.io/vcm-ml/fv3gfs-python",
    "cpu_count": 6,
    "gcp_secret": "gcp-key",
    "image_pull_policy": "Always",
}

RUNDIRS_DIRECTORY_NAME = "one_step_output"
CONFIG_DIRECTORY_NAME = "one_step_config"
VERTICAL_GRID_FILENAME = "fv_core.res.nc"
RESTART_CATEGORIES = ["fv_core.res", "fv_srf_wnd.res", "fv_tracer.res", "sfc_data"]
TILES = range(1, 7)


def get_model_config(config_update: dict) -> dict:
    """Get default model config and update with provided config_update"""
    config = fv3config.get_default_config()
    return _update_nested_dict(config, config_update)


def get_kubernetes_config(config_update: dict) -> dict:
    """Get default kubernetes config and update with provided config_update"""
    return _update_nested_dict(KUBERNETES_CONFIG_DEFAULT, config_update)


def _update_nested_dict(source_dict: dict, update_dict: dict) -> dict:
    for key, value in update_dict.items():
        if key in source_dict and isinstance(source_dict[key], Mapping):
            _update_nested_dict(source_dict[key], update_dict[key])
        else:
            source_dict[key] = update_dict[key]
    return source_dict


def _upload_if_necessary(path: str, bucket_url: str) -> str:
    if not path.startswith("gs://"):
        remote_path = os.path.join(bucket_url, os.path.basename(path))
        fsspec.filesystem("gs").put(path, remote_path)
        path = remote_path
    return path


def timestep_from_url(url: str) -> str:
    return str(Path(url).name)


def list_timesteps(bucket: str) -> set:
    """Returns the unique timesteps in a bucket"""
    fs = fsspec.filesystem("gs")
    file_list = fs.ls(bucket)
    timesteps = map(timestep_from_url, file_list)
    return set(timesteps)


def current_date_from_timestep(timestep: str) -> List[int]:
    """Return timestep in the format required by fv3gfs namelist"""
    year = int(timestep[:4])
    month = int(timestep[4:6])
    day = int(timestep[6:8])
    hour = int(timestep[9:11])
    minute = int(timestep[11:13])
    second = int(timestep[13:15])
    return [year, month, day, hour, minute, second]


def check_runs_complete(rundir_url: str):
    """Checks for existence of stdout.log and tail of said logfile to see if run
    successfully completed"""
    # TODO: Ideally this would check some sort of model exit code
    timestep_check_args = [
        (curr_timestep, os.path.join(rundir_url, curr_timestep, "stdout.log"))
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
    if fsspec.filesystem("gs").exists(logfile_path) and _check_log_tail(logfile_path):
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


def timesteps_to_process(input_bucket: str, output_bucket: str) -> List[str]:
    """Return list of timesteps left to process"""
    rundirs_url = os.path.join(output_bucket, RUNDIRS_DIRECTORY_NAME)
    to_do = list_timesteps(input_bucket)
    done = check_runs_complete(rundirs_url)
    logger.info(f"Number of input times: {len(to_do)}")
    logger.info(f"Number of completed times: {len(done)}")
    return sorted(list(set(to_do) - set(done)))


def _get_initial_condition_assets(input_bucket: str, timestep: str) -> List[dict]:
    """Get list of assets representing initial conditions for this timestep"""
    initial_condition_assets = [
        fv3config.get_asset_dict(
            os.path.join(input_bucket, timestep),
            f"{timestep}.{category}.tile{tile}.nc",
            target_location="INPUT",
            target_name=f"{category}.tile{tile}.nc",
        )
        for category in RESTART_CATEGORIES
        for tile in TILES
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


def _get_output_urls(url: str, timestep: str) -> tuple:
    """Return paths for config and rundir directories for this timestep"""
    config_url = os.path.join(url, CONFIG_DIRECTORY_NAME, timestep)
    rundir_url = os.path.join(url, RUNDIRS_DIRECTORY_NAME, timestep)
    return config_url, rundir_url


def _get_experiment_name(first_tag: str, second_tag: str) -> str:
    """Return unique experiment name with only alphanumeric, hyphen or dot"""
    uuid_short = str(uuid.uuid4())[:8]
    exp_name = f"one-step.{first_tag}.{second_tag}.{uuid_short}"
    return re.sub(r"[^a-zA-Z0-9.\-]", "", exp_name)


def _get_and_upload_config(
    workflow_name: str,
    one_step_config: dict,
    input_bucket: str,
    output_bucket: str,
    timestep: str,
) -> dict:
    """Update model and kubernetes configurations for this particular
    timestep and upload necessary files to GCS"""
    config_url, rundir_url = _get_output_urls(output_bucket, timestep)
    model_config = get_model_config(one_step_config["fv3config"])
    kubernetes_config = get_kubernetes_config(one_step_config["kubernetes"])
    model_config = fv3config.enable_restart(model_config)
    model_config["experiment_name"] = _get_experiment_name(workflow_name, timestep)
    model_config["initial_conditions"] = _get_initial_condition_assets(
        input_bucket, timestep
    )
    model_config["initial_conditions"].append(_get_vertical_grid_asset(config_url))
    model_config["namelist"]["coupler_nml"].update(
        {
            "current_date": current_date_from_timestep(timestep),
            "force_date_from_namelist": True,
        }
    )
    if kubernetes_config.get("runfile", None) is not None:
        kubernetes_config["runfile"] = _upload_if_necessary(
            kubernetes_config["runfile"], config_url
        )
    model_config["diag_table"] = _upload_if_necessary(
        model_config["diag_table"], config_url
    )
    fs = fsspec.filesystem("gs")
    fs.put(VERTICAL_GRID_FILENAME, os.path.join(config_url, VERTICAL_GRID_FILENAME))
    with fsspec.open(os.path.join(config_url, "fv3config.yml"), "w") as config_file:
        config_file.write(yaml.dump(model_config))
    return {"fv3config": model_config, "kubernetes": kubernetes_config}


def submit_jobs(
    timestep_list: List[str],
    workflow_name: str,
    one_step_config: dict,
    input_bucket: str,
    output_bucket: str,
):
    """Submit one-step job for all timesteps in timestep_list"""
    for timestep in timestep_list:
        one_step_config = _get_and_upload_config(
            workflow_name, one_step_config, input_bucket, output_bucket, timestep
        )
        jobname = one_step_config["fv3config"]["experiment_name"]
        one_step_config["kubernetes"]["jobname"] = jobname
        config_url, rundir_url = _get_output_urls(output_bucket, timestep)
        fv3config.run_kubernetes(
            os.path.join(config_url, "fv3config.yml"),
            rundir_url,
            **one_step_config["kubernetes"],
        )
        logger.info(f"Submitted job {jobname}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--one-step-yaml",
        type=str,
        required=True,
        help="Path to local run configuration yaml.",
    )
    parser.add_argument(
        "--input-bucket",
        type=str,
        required=True,
        help="Remote url to initial conditions. Initial conditions are assumed to be "
        "stored as INPUT_BUCKET/{timestamp}/{timestamp}.{restart_category}.tile*.nc",
    )
    parser.add_argument(
        "--output-bucket",
        type=str,
        required=True,
        help="Remote url where model configuration and output will be saved. "
        "Specifically, configuration files will be saved to OUTPUT_BUCKET/config "
        "and model output to OUTPUT_BUCKET/rundirs",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Number of timesteps to process. By default all timesteps "
        "found in INPUT-BUCKET will be processed. Useful for testing.",
    )
    args = parser.parse_args()
    with open(args.one_step_yaml) as file:
        one_step_config = yaml.load(file, Loader=yaml.FullLoader)
    workflow_name = os.path.splitext(args.one_step_yaml)[0]
    timestep_list = timesteps_to_process(args.input_bucket, args.output_bucket)
    timestep_list = timestep_list[: args.n_steps]
    logger.info(f"Number of times to process: {len(timestep_list)}")
    submit_jobs(
        timestep_list,
        workflow_name,
        one_step_config,
        args.input_bucket,
        args.output_bucket,
    )
