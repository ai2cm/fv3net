import argparse
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


def submit_job(bucket, run_config):
    model_config = get_model_config(run_config["fv3config"])
    job_name = model_config["experiment_name"] + f".{uuid.uuid4()}"
    kubernetes_config = get_kubernetes_config(run_config["kubernetes"])
    config_bucket = os.path.join(bucket, "config")
    with fsspec.open(os.path.join(config_bucket, "fv3config.yml"), "w") as config_file:
        config_file.write(yaml.dump(model_config))
    # if necessary, upload runfile and diag_table. In future, this should be
    # replaced with an fv3config function to do the same for all elements of config
    kubernetes_config["runfile"] = _upload_if_necessary(
        kubernetes_config["runfile"], config_bucket
    )
    model_config["diag_table"] = _upload_if_necessary(
        model_config["diag_table"], config_bucket
    )
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
        help="Remote url where config and output will be saved.",
    )
    parser.add_argument(
        "--run-yaml",
        type=str,
        required=True,
        help="Path to local run configuration yaml.",
    )
    args = parser.parse_args()
    with open(args.run_yaml) as file:
        run_config = yaml.load(file, Loader=yaml.FullLoader)
    submit_job(args.bucket, run_config)
