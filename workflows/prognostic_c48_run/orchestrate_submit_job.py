import argparse
import os
import yaml
import fsspec
import logging
from pathlib import Path

import fv3config
import fv3kube
from kubernetes.client import CoreV1Api, V1Pod, V1ObjectMeta

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
    "capture_output": False,
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
        "--fv3net-image",
        type=str,
        default="us.gcr.io/vcm-ml/fv3net:latest",
        help="fv3net docker image",
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


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()

    short_id = fv3kube.get_alphanumeric_unique_tag(8)
    job_label = {
        "orchestrator-jobs": f"prognostic-group-{short_id}",
        "app": "end-to-end",
    }

    # Get model config with prognostic run updates
    with open(args.prog_config_yml, "r") as f:
        prog_config_update = yaml.safe_load(f)
    model_config = fv3kube.get_full_config(
        prog_config_update, args.initial_condition_url, args.ic_timestep
    )

    kube_opts = KUBERNETES_DEFAULT.copy()
    if "kubernetes" in model_config:
        user_kube_config = model_config.pop("kubernetes")
        kube_opts.update(user_kube_config)

    config_dir = os.path.join(args.output_url, "job_config")
    job_config_path = os.path.join(config_dir, CONFIG_FILENAME)

    # try:
    #     diag_table = fv3kube.transfer_local_to_remote(
    #         model_config["diag_table"], config_dir
    #     )
    # except FileNotFoundError:
    #     diag_table = model_config["diag_table"]

    model_config[
        "diag_table"
    ] = "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic"

    # Add scikit learn ML model config section
    scikit_learn_config = {}
    scikit_learn_config["zarr_output"] = "diags.zarr"
    if args.model_url:

        # insert the model asset
        model_url = os.path.join(args.model_url, MODEL_FILENAME)
        model_asset = fv3config.get_asset_dict(
            args.model_url, MODEL_FILENAME, target_name="model.pkl"
        )
        model_config.setdefault("patch_files", []).append(model_asset)

        scikit_learn_config.update(
            model="model.pkl", diagnostic_ml=args.diagnostic_ml,
        )
        # TODO uncomment or delete
        # kube_opts["runfile"] = fv3kube.transfer_local_to_remote(RUNFILE, config_dir)

    model_config["scikit_learn"] = scikit_learn_config

    # Upload the new prognostic config
    with fsspec.open(job_config_path, "w") as f:
        f.write(yaml.dump(model_config))

    # need to initialized the client differently than
    # run_kubernetes when running in a pod.

    # TODO need another documented iterative development paradigm if not uploading
    # runfile

    fv3kube.load_kube_config()
    client = CoreV1Api()

    pod = V1Pod(
        metadata=V1ObjectMeta(generate_name="prognostic-run-", labels=job_label),
        spec=fv3kube.containers.post_processed_fv3_pod_spec(
            model_config,
            args.output_url,
            fv3config_image=args.docker_image,
            fv3_image=args.docker_image,
            post_process_image=args.fv3net_image,
        ),
    )

    created_pod = client.create_namespaced_pod("default", body=pod)
    print(created_pod.metadata.name)

    if not args.detach:
        fv3kube.wait_for_complete(job_label, raise_on_fail=not args.allow_fail)
        fv3kube.delete_completed_jobs(job_label)
