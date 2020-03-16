import argparse
import os
import yaml
import fsspec
import logging
from pathlib import Path

import fv3config
import fv3net.pipelines.kube_jobs as kube_jobs
from vcm.cloud.fsspec import get_fs
from fv3net.pipelines.common import get_alphanumeric_unique_tag

logger = logging.getLogger(__name__)
PWD = Path(os.path.abspath(__file__)).parent
RUNFILE = os.path.join(PWD, "sklearn_runfile.py")
CONFIG_FILENAME = "fv3config.yml"
MODEL_FILENAME = "sklearn_model.pkl"

KUBERNETES_DEFAULT = {
    "cpu_count": 6,
    "memory_gb": 3.6,
    "gcp_secret": "gcp-key",
    "image_pull_policy": "Always",
}


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "initial_condition_url",
        type=str,
        help="Remote url to directory holding timesteps with model initial conditions.",
    )
    parser.add_argument(
        "ic_timestep",
        type=str,
        help="Time step to grab from the initial conditions url.",
    )
    parser.add_argument(
        "docker_image",
        type=str,
        help="Docker image to pull for the prognostic run kubernetes pod.",
    )
    parser.add_argument(
        "output_url",
        type=str,
        help="Remote storage location for prognostic run output.",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default=None,
        help="Remote url to a trained sklearn model.",
    )
    parser.add_argument(
        "--prog_config_yml",
        type=str,
        default="prognostic_config.yml",
        help="Path to a config update YAML file specifying the changes (e.g., "
        "diag_table, runtime, ...) from the one-step runs for the prognostic run.",
    )
    parser.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="Do not wait for the k8s job to complete.",
    )

    return parser


def _get_onestep_config(restart_path, timestep):

    config_path = os.path.join(restart_path, timestep, CONFIG_FILENAME)
    fs = get_fs(config_path)
    with fs.open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def _update_with_prognostic_model_config(model_config, prognostic_config):
    """
    Update the default model config with the prognostic run.
    """

    with open(prognostic_config, "r") as f:
        prognostic_model_config = yaml.load(f, Loader=yaml.FullLoader)

    kube_jobs.update_nested_dict(model_config, prognostic_model_config)

    return model_config


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()

    short_id = get_alphanumeric_unique_tag(8)
    job_name = f"prognostic-run-{short_id}"
    job_label = {"orchestrator-jobs": f"prognostic-group-{short_id}"}

    model_config = _get_onestep_config(args.initial_condition_url, args.ic_timestep)

    # Get model config with one-step and prognistic run updates
    model_config = _update_with_prognostic_model_config(
        model_config, args.prog_config_yml
    )
    print(model_config)

    kube_opts = KUBERNETES_DEFAULT.copy()
    if "kubernetes" in model_config:
        user_kube_config = model_config.pop("kubernetes")
        kube_opts.update(user_kube_config)

    job_config_filename = "fv3config.yml"
    config_dir = os.path.join(args.output_url, "job_config")
    job_config_path = os.path.join(config_dir, CONFIG_FILENAME)

    model_config["diag_table"] = kube_jobs.transfer_local_to_remote(
        model_config["diag_table"], config_dir
    )

    # Add prognostic config section
    if args.model_url:
        model_config["scikit_learn"] = {
            "model": os.path.join(args.model_url, MODEL_FILENAME),
            "zarr_output": "diags.zarr",
        }
        kube_opts["runfile"] = kube_jobs.transfer_local_to_remote(RUNFILE, config_dir)

    # Upload the new prognostic config
    with fsspec.open(job_config_path, "w") as f:
        f.write(yaml.dump(model_config))

    fv3config.run_kubernetes(
        config_location=job_config_path,
        outdir=args.output_url,
        jobname=job_name,
        docker_image=args.docker_image,
        job_labels=job_label,
        **kube_opts,
    )

    if not args.detach:
        successful, _ = kube_jobs.wait_for_complete(job_label)
        kube_jobs.delete_job_pods(successful)
