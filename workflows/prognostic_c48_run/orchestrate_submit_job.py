from typing import Sequence, Optional
import argparse
import os
import yaml
import fsspec
import logging
from pathlib import Path
import json

import fv3config
import fv3kube
from kubernetes.client import (
    CoreV1Api,
    V1ResourceRequirements,
    V1Pod,
    V1Toleration,
    V1ObjectMeta,
    V1PodSpec,
    V1Container,
    V1VolumeMount,
    V1Volume,
    V1SecretVolumeSource,
    V1EmptyDirVolumeSource,
    V1EnvVar,
)

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

script_template = """
import fv3config
import yaml
import os

model_url = os.environ["MODEL"]
config = yaml.safe_load(os.environ["FV3CONFIG"])
fv3config.write_run_directory(config, "{rundir}")
with open("{rundir}/fv3config.yml", "w") as f:
    yaml.safe_dump(config, f)
"""


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


def insert_gcp_secret(container: V1Container, secret_vol: V1Volume):
    """Insert GCP credentials into a container

    Args:
        container: a container to insert the gcp credentials into
        secret_vol: a volume containg a GCP service account key at
            ``key.json`` 
            
    """
    if container.volume_mounts is None:
        container.volume_mounts = []

    if container.env is None:
        container.env = []

    container.volume_mounts.append(
        V1VolumeMount(mount_path="/etc/gcp/", name=secret_vol.name)
    )

    container.env.extend(
        [
            V1EnvVar(name="GOOGLE_APPLICATION_CREDENTIALS", value="/etc/gcp/key.json"),
            V1EnvVar(
                name="CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", value="/etc/gcp/key.json"
            ),
        ]
    )


def write_rundir_container(
    config, empty_vol: V1Volume, secret_vol: V1Volume
) -> V1Container:
    # TODO Refactor this to the __main__ of an fv3config container
    # this could be on dockerhub

    container = V1Container(name="write-rundir")
    container.volume_mounts = [
        V1VolumeMount(mount_path="/mnt", name=empty_vol.name),
    ]

    # Inject the fv3config and runfile via an environmental variable
    container.env = [
        V1EnvVar(name="FV3CONFIG", value=yaml.safe_dump(config)),
        V1EnvVar(name="MODEL", value=yaml.safe_dump(config)),
    ]
    container.image = args.docker_image
    container.command = ["python", "-c"]
    container.args = [script_template.format(rundir="/mnt/rundir")]

    insert_gcp_secret(container, secret_vol)

    return container


def fv3_container(empty_vol: V1Volume) -> V1Container:
    """

    Args:

    Returns:
        container for running the output. stores data in the volume at path "rundir".
    """
    runfile_path = "/fv3net/workflows/prognostic_c48_run/sklearn_runfile.py"
    fv3_container = V1Container(name="fv3")
    fv3_container.image = args.docker_image
    fv3_container.working_dir = "/mnt/rundir"
    fv3_container.command = [
        "mpirun",
        "-np",
        str(6),
        "--oversubscribe",
        "--allow-run-as-root",
        "--mca",
        "btl_vader_single_copy_mechanism",
        "none",
        "python",
        runfile_path,
    ]
    # Suitable for C48 job
    fv3_container.resources = V1ResourceRequirements(
        limits=dict(cpu="6", memory="6Gi"), requests=dict(cpu="6", memory="6Gi"),
    )

    fv3_container.volume_mounts = [
        V1VolumeMount(mount_path="/mnt", name=empty_vol.name),
    ]

    return fv3_container


def post_process_container(
    path: str, destination: str, data_vol: V1Volume, secret_vol: V1Volume, image: str
) -> V1Container:
    """Container for post processing fv3 model output for cloud storage

    Args:
        path: relative path within volume pointing to run-directory
        destination: uri to desired GCS output directory
        vol: a K8s volume containing ``path`
        image: the docker image to use for post processing. Needs to contain
            the fv3net source tree at the path "/fv3net".
    Returns
        a k8s container encapsulating the post-processing.
    """
    container = V1Container(name="post")
    container.volume_mounts = [
        V1VolumeMount(mount_path="/mnt", name=data_vol.name),
    ]

    rundir = os.path.join("/mnt", path)
    container.image = image
    container.working_dir = "/fv3net/workflows/prognostic_c48_run"
    container.command = ["python", "post_process.py", rundir, destination]
    # Suitable for C48 job
    container.resources = V1ResourceRequirements(
        limits=dict(cpu="6", memory="3600M"), requests=dict(cpu="6", memory="3600M"),
    )

    insert_gcp_secret(container, secret_vol)

    return container


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

    empty_vol = V1Volume(name="rundir")
    empty_vol.empty_dir = V1EmptyDirVolumeSource()

    secret_vol = V1Volume(name="google-secret")
    secret_vol.secret = V1SecretVolumeSource(secret_name="gcp-key")

    # Need to add toleration for large jobs
    climate_sim_toleration = V1Toleration(
        effect="NoSchedule", value="climate-sim-pool", key="dedicated", operator="Equal"
    )

    pod = V1Pod()
    pod.spec = V1PodSpec(
        init_containers=[
            write_rundir_container(model_config, empty_vol, secret_vol),
            fv3_container(empty_vol),
        ],
        containers=[
            post_process_container(
                "rundir", args.output_url, empty_vol, secret_vol, args.docker_image
            )
        ],
        volumes=[empty_vol, secret_vol],
        restart_policy="Never",
        tolerations=[climate_sim_toleration],
    )
    pod.metadata = V1ObjectMeta(generate_name="prognostic-run-", labels=job_label)
    created_pod = client.create_namespaced_pod("default", body=pod)
    print(created_pod.metadata.name)

    if not args.detach:
        fv3kube.wait_for_complete(job_label, raise_on_fail=not args.allow_fail)
        fv3kube.delete_completed_jobs(job_label)
