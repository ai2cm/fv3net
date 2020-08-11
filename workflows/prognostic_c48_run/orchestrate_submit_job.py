import argparse
import os
import yaml
import logging
from pathlib import Path
import copy

import fv3config
import fv3kube

logger = logging.getLogger(__name__)
PWD = Path(os.path.abspath(__file__)).parent
RUNFILE = os.path.join(PWD, "sklearn_runfile.py")
CONFIG_FILENAME = "fv3config.yml"


def get_kube_opts(config_update, image_tag=None):
    default = {
        "gcp_secret_name": "gcp-key",
        "post_process_image": "us.gcr.io/vcm-ml/post_process_run",
        "fv3_image": "us.gcr.io/vcm-ml/prognostic_run",
        "fv3config_image": "us.gcr.io/vcm-ml/prognostic_run",
    }
    kube_opts = copy.copy(default)
    kube_opts.update(config_update.get("kubernetes", {}))
    if args.image_tag:
        for key in ["fv3config_image", "fv3_image", "post_process_image"]:
            kube_opts[key] += ":" + args.image_tag
    return kube_opts


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
        "--image-tag", type=str, default=None, help="tag to apply to all default images"
    )
    parser.add_argument(
        "--prog_config_yml",
        type=str,
        default="prognostic_config.yml",
        help="Path to a config update YAML file specifying the changes from the base"
        "fv3config (e.g. diag_table, runtime, ...) for the prognostic run.",
    )
    parser.add_argument(
        "--diagnostic_ml",
        action="store_true",
        help="Compute and save ML predictions but do not apply them to model state.",
    )
    parser.add_argument(
        "-a",
        "--allow_fail",
        action="store_true",
        help="Do not raise error if job fails.",
    )
    parser.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="Do not wait for the k8s job to complete.",
    )

    return parser


def _update_config_for_ml(
    model_config: dict, model_url: str, diagnostic_ml: bool
) -> None:
    """update various entries needed to ML, helper function refactored from __main__
    """

    # TODO rename hardcoded references to sklearn?

    scikit_learn_config = model_config.get("scikit_learn", {})
    scikit_learn_config["zarr_output"] = "diags.zarr"
    model_config.update(scikit_learn=scikit_learn_config)

    if model_url is not None:
        # insert the model asset
        model_type = scikit_learn_config.get("model_type", "scikit_learn")
        if model_type == "scikit_learn":
            _update_sklearn_config(model_config, model_url)
        elif model_type == "keras":
            _update_keras_config(model_config, model_url)
        else:
            raise ValueError(
                "Available model types are scikit_learn and keras; received type:"
                f" {model_type}."
            )
        model_config["scikit_learn"].update(diagnostic_ml=diagnostic_ml)


def _update_sklearn_config(
    model_config: dict, model_url: str, sklearn_filename: str = "sklearn_model.pkl"
) -> None:
    model_asset = fv3config.get_asset_dict(
        model_url, sklearn_filename, target_name=sklearn_filename
    )
    model_config.setdefault("patch_files", []).append(model_asset)
    model_config["scikit_learn"].update(model=sklearn_filename)


def _update_keras_config(
    model_config: dict, model_url: str, keras_dirname: str = "model_data"
) -> None:
    model_asset_list = fv3config.asset_list_from_path(
        os.path.join(args.model_url, keras_dirname), target_location=keras_dirname
    )
    model_config.setdefault("patch_files", []).extend(model_asset_list)
    model_config["scikit_learn"].update(model=keras_dirname)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()

    # Get model config with prognostic run updates
    with open(args.prog_config_yml, "r") as f:
        prog_config_update = yaml.safe_load(f)
    config_dir = os.path.join(args.output_url, "job_config")
    job_config_path = os.path.join(config_dir, CONFIG_FILENAME)

    short_id = fv3kube.get_alphanumeric_unique_tag(8)
    job_label = {
        "orchestrator-jobs": f"prognostic-group-{short_id}",
        # needed to use pod-disruption budget
        "app": "end-to-end",
    }
    model_config = fv3kube.get_full_config(
        prog_config_update, args.initial_condition_url, args.ic_timestep
    )

    # hard-code the diag_table
    model_config[
        "diag_table"
    ] = "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic"

    # Add ML model config section
    _update_config_for_ml(model_config, args.model_url, args.diagnostic_ml)

    kube_opts = get_kube_opts(prog_config_update, args.image_tag)
    pod_spec = fv3kube.containers.post_processed_fv3_pod_spec(
        model_config, args.output_url, **kube_opts
    )
    job = fv3kube.containers.pod_spec_to_job(
        pod_spec, labels=job_label, generate_name="prognostic-run-"
    )

    client = fv3kube.initialize_batch_client()
    created_job = client.create_namespaced_job("default", job)
    logger.info(f"Created job: {created_job.metadata.name}")

    if not args.detach:
        fv3kube.wait_for_complete(job_label, raise_on_fail=not args.allow_fail)
        fv3kube.delete_completed_jobs(job_label)
