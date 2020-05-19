import argparse
import logging
import os
import fsspec
import yaml
import fv3config
import fv3kube
import vcm

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


def get_kubernetes_config(config_update):
    """Get default kubernetes config and update with provided config_update"""
    return vcm.update_nested_dict(KUBERNETES_CONFIG_DEFAULT, config_update)


def _get_and_upload_run_config(bucket, run_config, base_model_config):
    """Get config objects for current job and upload as necessary"""
    config_bucket = os.path.join(bucket, "config")
    model_config = vcm.update_nested_dict(
        base_model_config, run_config["fv3config"]
    )
    kubernetes_config = get_kubernetes_config(run_config["kubernetes"])
    # if necessary, upload runfile and diag_table. In future, this should be
    # replaced with an fv3config function to do the same for all elements of config
    if kubernetes_config["runfile"] is not None:
        kubernetes_config["runfile"] = fv3kube.transfer_local_to_remote(
            kubernetes_config["runfile"], config_bucket
        )
    model_config["diag_table"] = fv3kube.transfer_local_to_remote(
        model_config["diag_table"], config_bucket
    )
    if model_config["namelist"]["fv_core_nml"].get("nudge", False):
        model_config = fv3kube.update_config_for_nudging(model_config, config_bucket)
    with fsspec.open(os.path.join(config_bucket, "fv3config.yml"), "w") as config_file:
        config_file.write(yaml.dump(model_config))
    return {"kubernetes": kubernetes_config, "fv3config": model_config}, config_bucket


def submit_job(bucket, run_config, base_model_config):
    run_config, config_bucket = _get_and_upload_run_config(
        bucket, run_config, base_model_config
    )
    tag = fv3kube.get_alphanumeric_unique_tag(8)
    job_name = run_config["fv3config"]["experiment_name"] + f"-{tag}"
    run_config["kubernetes"]["jobname"] = job_name
    fv3config.run_kubernetes(
        os.path.join(config_bucket, "fv3config.yml"),
        os.path.join(bucket, "output"),
        **run_config["kubernetes"],
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
        default="v0.3",
        help="Default fv3config.yml version to use as the base configuration. "
        "This should be consistent with the fv3gfs-python version in the specified "
        "docker image.",
    )
    args, extra_args = parser.parse_known_args()
    base_model_config = fv3kube.get_base_fv3config(args.config_version)
    with open(args.run_yaml) as file:
        run_config = yaml.load(file, Loader=yaml.FullLoader)
    submit_job(args.bucket, run_config, base_model_config)
