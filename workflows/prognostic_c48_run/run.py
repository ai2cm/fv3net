import argparse
import os
import yaml
import fsspec
import logging
from pathlib import Path
import tempfile

import fv3config
import fv3kube
import vcm
from vcm.cloud import get_fs

logger = logging.getLogger(__name__)
PWD = Path(os.path.abspath(__file__)).parent
RUNFILE = os.path.join(PWD, "sklearn_runfile.py")
CONFIG_FILENAME = "fv3config.yml"
MODEL_FILENAME = "sklearn_model.pkl"


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
    return parser


def _get_onestep_config(restart_path, timestep):
    config_path = os.path.join(
        restart_path, "one_step_config", timestep, CONFIG_FILENAME
    )
    fs = get_fs(config_path)
    with fs.open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def _update_with_prognostic_model_config(model_config, prognostic_config):
    """
    Update the default model config with the prognostic run.
    """

    with open(prognostic_config, "r") as f:
        prognostic_model_config = yaml.safe_load(f)

    vcm.update_nested_dict(model_config, prognostic_model_config)

    return model_config


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()

    short_id = fv3kube.get_alphanumeric_unique_tag(8)
    job_name = f"prognostic-run-{short_id}"
    job_label = {
        "orchestrator-jobs": f"prognostic-group-{short_id}",
        "app": "end-to-end",
    }

    model_config = _get_onestep_config(args.initial_condition_url, args.ic_timestep)

    # Get model config with one-step and prognistic run updates
    model_config = _update_with_prognostic_model_config(
        model_config, args.prog_config_yml
    )

    job_config_filename = "fv3config.yml"
    config_dir = os.path.join(args.output_url, "job_config")
    job_config_path = os.path.join(config_dir, CONFIG_FILENAME)

    model_config["diag_table"] = fv3kube.transfer_local_to_remote(
        model_config["diag_table"], config_dir
    )

    # Add prognostic config section
    if args.model_url:
        scikit_learn_config = model_config.get("scikit_learn", {})
        scikit_learn_config.update(
            model=os.path.join(args.model_url, MODEL_FILENAME),
            zarr_output="diags.zarr",
        )
        model_config["scikit_learn"] = scikit_learn_config
        kube_opts["runfile"] = fv3kube.transfer_local_to_remote(RUNFILE, config_dir)

    # Upload the new prognostic config
    with fsspec.open(job_config_path, "w") as f:
        f.write(yaml.dump(model_config))

    # need to initialized the client differently than
    # run_kubernetes when running in a pod.
    client = fv3kube.initialize_batch_client()
    job = fv3config.run_kubernetes(
        config_location=job_config_path,
        outdir=args.output_url,
        jobname=job_name,
        docker_image=args.docker_image,
        job_labels=job_label,
        submit=False,
        **kube_opts,
    )
    client.create_namespaced_job(namespace="default", body=job)

    if not args.detach:
        fv3kube.wait_for_complete(job_label)
        fv3kube.delete_completed_jobs(job_label)
