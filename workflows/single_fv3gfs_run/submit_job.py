import argparse
from datetime import datetime, timedelta
import logging
import os
import uuid
import fsspec
import yaml
import fv3config

logger = logging.getLogger("run_jobs")

KUBERNETES_CONFIG_DEFAULT = {
    "docker_image": "us.gcr.io/vcm-ml/fv3gfs-python",
    "runfile": "runfile.py",
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


def get_model_config(config_update) -> dict:
    config = fv3config.get_default_config()
    return _update_dict(config, config_update)


def get_kubernetes_config(config_update) -> dict:
    return _update_dict(KUBERNETES_CONFIG_DEFAULT, config_update)


def _update_dict(source_dict, update_dict):
    for key, value in update_dict.items():
        if key in source_dict and isinstance(source_dict[key], dict):
            _update_dict(source_dict[key], update_dict[key])
        else:
            source_dict[key] = update_dict[key]
    return source_dict


def _upload_if_necessary(path, bucket_url):
    if not path.startswith("gs://"):
        remote_path = os.path.join(bucket_url, os.path.basename(path))
        fsspec.filesystem("gs").put(path, remote_path)
        path = remote_path
    return path


def get_nudge_file_list(start_date: datetime, run_duration: timedelta) -> list:
    run_duration_hours = run_duration.days * HOURS_IN_DAY
    nudging_hours = range(0, run_duration_hours + NUDGE_INTERVAL, NUDGE_INTERVAL)
    time_list = [start_date + timedelta(hours=hour) for hour in nudging_hours]
    return [time.strftime(NUDGE_FILENAME_PATTERN) for time in time_list]


def _get_0z_start_date(config):
    """ Return datetime object for 00:00:00 on current_date """
    current_date = config["namelist"]["coupler_nml"]["current_date"]
    return datetime(current_date[0], current_date[1], current_date[2])


def get_nudge_files_asset_list(config):
    start_date = _get_0z_start_date(config)
    run_duration = fv3config.get_run_duration(config)
    return [
        fv3config.get_asset_dict(NUDGE_BUCKET, file, target_location="INPUT")
        for file in get_nudge_file_list(start_date, run_duration)
    ]


def get_and_write_input_fname_asset(config, config_bucket):
    input_fname_list = config["namelist"]["fv_nwp_nudge_nml"]["input_fname_list"]
    start_date = _get_0z_start_date(config)
    run_duration = fv3config.get_run_duration(config)
    with fsspec.open(os.path.join(config_bucket, input_fname_list), "w") as remote_file:
        remote_file.write("\n".join(get_nudge_file_list(start_date, run_duration)))
    return fv3config.get_asset_dict(config_bucket, input_fname_list)


def update_config_for_nudging(model_config, config_bucket):
    model_config["patch_files"] = get_nudge_files_asset_list(model_config)
    model_config["patch_files"].append(
        get_and_write_input_fname_asset(model_config, config_bucket)
    )
    return model_config


def submit_job(bucket, run_config):
    config_bucket = os.path.join(bucket, "config")
    model_config = get_model_config(run_config["fv3config"])
    kubernetes_config = get_kubernetes_config(run_config["kubernetes"])
    # if necessary, upload runfile and diag_table. In future, this should be
    # replaced with an fv3config function to do the same for all elements of config
    kubernetes_config["runfile"] = _upload_if_necessary(
        kubernetes_config["runfile"], config_bucket
    )
    model_config["diag_table"] = _upload_if_necessary(
        model_config["diag_table"], config_bucket
    )
    if model_config["namelist"]["fv_core_nml"].get("nudge", False):
        model_config = update_config_for_nudging(model_config, config_bucket)
    with fsspec.open(os.path.join(config_bucket, "fv3config.yml"), "w") as config_file:
        config_file.write(yaml.dump(model_config))
    job_name = model_config["experiment_name"] + f".{uuid.uuid4()}"
    fv3config.run_kubernetes(
        os.path.join(config_bucket, "fv3config.yml"),
        os.path.join(bucket, "output"),
        kubernetes_config["docker_image"],
        runfile=kubernetes_config["runfile"],
        jobname=job_name,
        namespace=kubernetes_config["namespace"],
        memory_gb=kubernetes_config["memory_gb"],
        cpu_count=kubernetes_config["cpu_count"],
        gcp_secret=kubernetes_config["gcp_secret"],
        image_pull_policy=kubernetes_config["image_pull_policy"],
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
    args, extra_args = parser.parse_known_args()
    with open(args.run_yaml) as file:
        run_config = yaml.load(file, Loader=yaml.FullLoader)
    submit_job(args.bucket, run_config)
