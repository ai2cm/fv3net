import argparse
from collections.abc import Mapping
from datetime import datetime, timedelta
import logging
import os
import uuid
import fsspec
import yaml
import fv3config
from fv3net.pipelines.kube_jobs import get_base_fv3config

logger = logging.getLogger("run_jobs")

KUBERNETES_CONFIG_DEFAULT = {
    "docker_image": "us.gcr.io/vcm-ml/fv3gfs-python",
    "runfile": None,
    "namespace": "default",
    "memory_gb": 3.6,
    "cpu_count": 6,
    "gcp_secret": "gcp-key",
    "image_pull_policy": "Always",
}

HOURS_IN_DAY = 24
NUDGE_INTERVAL = 6  # hours
NUDGE_FILENAME_PATTERN = "%Y%m%d_%HZ_T85LR.nc"
NUDGE_BUCKET = "gs://vcm-ml-data/2019-12-02-year-2016-T85-nudging-data"


def get_model_config(config_update):
    """Get default model config and update with provided config_update"""
    config = fv3config.get_default_config()
    return _update_nested_dict(config, config_update)


def get_kubernetes_config(config_update):
    """Get default kubernetes config and updatedwith provided config_update"""
    return _update_nested_dict(KUBERNETES_CONFIG_DEFAULT, config_update)


def _update_nested_dict(source_dict, update_dict):
    for key, value in update_dict.items():
        if key in source_dict and isinstance(source_dict[key], Mapping):
            _update_nested_dict(source_dict[key], update_dict[key])
        else:
            source_dict[key] = update_dict[key]
    return source_dict


def _upload_if_necessary(path, bucket_url):
    if not path.startswith("gs://"):
        remote_path = os.path.join(bucket_url, os.path.basename(path))
        fsspec.filesystem("gs").put(path, remote_path)
        path = remote_path
    return path


def _get_nudge_file_list(config):
    """Return python list of filenames of all nudging files required"""
    start_date = _get_0z_start_date(config)
    run_duration = fv3config.get_run_duration(config)
    run_duration_hours = run_duration.days * HOURS_IN_DAY
    nudging_hours = range(0, run_duration_hours + NUDGE_INTERVAL, NUDGE_INTERVAL)
    time_list = [start_date + timedelta(hours=hour) for hour in nudging_hours]
    return [time.strftime(NUDGE_FILENAME_PATTERN) for time in time_list]


def _get_0z_start_date(config):
    """Return datetime object for 00:00:00 on current_date"""
    current_date = config["namelist"]["coupler_nml"]["current_date"]
    return datetime(current_date[0], current_date[1], current_date[2])


def _get_nudge_files_asset_list(config):
    """Return list of fv3config assets for all nudging files required"""
    return [
        fv3config.get_asset_dict(NUDGE_BUCKET, file, target_location="INPUT")
        for file in _get_nudge_file_list(config)
    ]


def _get_and_write_nudge_files_description_asset(config, config_bucket):
    """Write a text file with list of all nudging files required  (which the
    model requires to know what the nudging files are called) and return an fv3config
    asset pointing to this text file."""
    input_fname_list = config["namelist"]["fv_nwp_nudge_nml"]["input_fname_list"]
    with fsspec.open(os.path.join(config_bucket, input_fname_list), "w") as remote_file:
        remote_file.write("\n".join(_get_nudge_file_list(config)))
    return fv3config.get_asset_dict(config_bucket, input_fname_list)


def _update_config_for_nudging(model_config, config_bucket):
    """Add assets to config for all nudging files and for the text file
    listing nudging files"""
    if "patch_files" not in model_config:
        model_config["patch_files"] = []
    model_config["patch_files"].append(
        _get_and_write_nudge_files_description_asset(model_config, config_bucket)
    )
    model_config["patch_files"].extend(_get_nudge_files_asset_list(model_config))
    return model_config


def _get_and_upload_run_config(bucket, run_config, base_model_config):
    """Get config objects for current job and upload as necessary"""
    config_bucket = os.path.join(bucket, "config")
    model_config = _update_nested_dict(base_model_config, run_config["fv3config"])
    kubernetes_config = get_kubernetes_config(run_config["kubernetes"])
    # if necessary, upload runfile and diag_table. In future, this should be
    # replaced with an fv3config function to do the same for all elements of config
    if kubernetes_config["runfile"] is not None:
        kubernetes_config["runfile"] = _upload_if_necessary(
            kubernetes_config["runfile"], config_bucket
        )
    model_config["diag_table"] = _upload_if_necessary(
        model_config["diag_table"], config_bucket
    )
    if model_config["namelist"]["fv_core_nml"].get("nudge", False):
        model_config = _update_config_for_nudging(model_config, config_bucket)
    with fsspec.open(os.path.join(config_bucket, "fv3config.yml"), "w") as config_file:
        config_file.write(yaml.dump(model_config))
    return {"kubernetes": kubernetes_config, "fv3config": model_config}, config_bucket


def submit_job(bucket, run_config, base_model_config):
    run_config, config_bucket = _get_and_upload_run_config(
        bucket, run_config, base_model_config
    )
    job_name = run_config["fv3config"]["experiment_name"] + f".{uuid.uuid4()}"
    fv3config.run_kubernetes(
        os.path.join(config_bucket, "fv3config.yml"),
        os.path.join(bucket, "output"),
        run_config["kubernetes"]["docker_image"],
        runfile=run_config["kubernetes"]["runfile"],
        jobname=job_name,
        namespace=run_config["kubernetes"]["namespace"],
        memory_gb=run_config["kubernetes"]["memory_gb"],
        cpu_count=run_config["kubernetes"]["cpu_count"],
        gcp_secret=run_config["kubernetes"]["gcp_secret"],
        image_pull_policy=run_config["kubernetes"]["image_pull_policy"],
    )
    logger.info(f"Submitted {job_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="Remote url where config and output will be saved. Specifically, "
        "configuration will be saved to BUCKET/config and output to BUCKET/output",
    )
    parser.add_argument(
        "--run-yaml",
        type=str,
        required=True,
        help="Path to local run configuration yaml.",
    )
    parser.add_argument(
        "--config-version",
        type=str,
        required=False,
        default="v0.2",
        help="Default fv3config.yml version to use as the base configuration. "
        "This should be consistent with the fv3gfs-python version in the specified "
        "docker image.",
    )
    args, extra_args = parser.parse_known_args()
    base_model_config = get_base_fv3config(args.config_version)
    with open(args.run_yaml) as file:
        run_config = yaml.load(file, Loader=yaml.FullLoader)
    submit_job(args.bucket, run_config, base_model_config)
