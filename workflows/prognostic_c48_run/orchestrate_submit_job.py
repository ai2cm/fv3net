import argparse
import uuid
import os
import yaml
from pathlib import Path

import fv3config
import fv3net.pipelines.kube_jobs.utils as kubejob_utils

PWD = Path(__file__).parent
RUNFILE = os.path.join(PWD, "orchestrator_runfile.py")

fv3config.run_kubernetes(
    os.path.join(curr_config_url, "fv3config.yml"),
    curr_output_url,
    experiment_label=experiment_label,
    **one_step_config["kubernetes"],
)


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_url",
        type=str,
        help="Remote url to a trained sklearn model.",
    )
    parser.add_argument(
        "initial_condition_url",
        type=str,
        help="Remote url to directory holding timesteps with model initial conditions."
    )
    parser.add_argument(
        "output_url",
        type=str,
        help="Remote storage location for prognostic run output."
    )
    parser.add_argument(
        "base_fv3config_yml",
        type=str,
        help="Path to fv3config YAML file used to perform the one-step training data runs."
    )
    parser.add_argument(
        "prog_config_yml",
        type=str,
        help="Path to a config update YAML file specifying the changes (e.g., diag_table, runtime, ...) from the one-step runs for the prognostic run.",
    )
    parser.add_argument(
        "ic_timestep",
        type=str,
        required=False,
        help="Time step to grab from the initial conditions url.,
    )


def _get_prognostic_model_config(base_config, prognostic_config):
    """
    Update the default model config with first the one-step job config
    and then updates for the the prognostic run.
    """

    model_config = fv3config.get_default_config()
    with open(base_config, "r") as f:
        fv3config = yaml.load(f, Loader=yaml.FullLoader)
        onestep_model_config = fv3config["fv3config"]
    with open(prognostic_config, "r") as f:
        prognostic_model_config = yaml.load(f, Loader=yaml.FullLoader)

    kubejob_utils.update_nested_dict(model_config, onestep_model_config)
    kubejob_utils.update_nested_dict(model_config, prognostic_model_config)

    return model_config


if __name__ == "__main__":

    parser = _create_arg_parser()
    args = parser.parse_args()

    short_id = uuid.uuid4()[-8:]
    job_name = f"prognostic-run-{short_id}"

    # Get model config with one-step and prognistic run updates
    model_config = _get_prognostic_model_config(
        args.base_fv3config_yml, args.prog_config_yml
    )

    config_url = os.path.join(args.output_url, "job_config")
    job_config_filename = "fv3config.yml"

    # Add prognostic config section
    model_config["scikit_learn"] = {
        "model": args.model_url,
        "zarr_output": os.path.join(args.output_url, "diags.zarr")
    }

    # Prepare remote run directory
    model_config = kubejob_utils.prepare_and_upload_model_config(
        job_name, args.input_url, config_url, args.timestep, model_config,
        upload_config_filename=job_config_filename,
    )

    remote_runfile_path = kubejob_utils.upload_runfile(
        {"runfile": RUNFILE}
    )
    job_config_path = os.path.join(config_url, job_config_filename)

    fv3config.run_kubernetes(
        config_location=job_config_path,
        outdir=args.output_url,
        docker_image="us.gcr.io/vcm-ml/fv3gfs-python:v0.2.1",
        runfile=remote_runfile_path,
        cpu_count=6,
        gcp_secret="gcp-key",
        image_pull_policy="Always",
    )
