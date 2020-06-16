import os
import yaml
from typing import Mapping

from kubernetes.client import (
    V1ResourceRequirements,
    V1Toleration,
    V1PodSpec,
    V1Container,
    V1VolumeMount,
    V1Volume,
    V1SecretVolumeSource,
    V1EmptyDirVolumeSource,
    V1EnvVar,
)


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


def write_rundir_container(
    config, data_vol: V1Volume, secret_vol: V1Volume, image: str
) -> V1Container:
    """Container for writing run-directories with fv3config

    Args:
        config: an fv3config dictionary
        data_vol: a k8s volume to store the data in. The data will be written
            to the location "rundir" in this volume.
        secret_vol: a volume contain a GCP service account key at the path
            ``key.json``
        image: an docker image with fv3config installed
    Returns:
        a k8s container that will write the run directory
        
    """
    # TODO Refactor this to the __main__ of an fv3config container
    # this could be on dockerhub

    container = V1Container(name="write-rundir")
    container.volume_mounts = [
        V1VolumeMount(mount_path="/mnt", name=data_vol.name),
    ]

    # Inject the fv3config and runfile via an environmental variable
    container.env = [
        V1EnvVar(name="FV3CONFIG", value=yaml.safe_dump(config)),
        V1EnvVar(name="MODEL", value=yaml.safe_dump(config)),
    ]
    container.image = image
    container.command = ["python", "-c"]
    container.args = [script_template.format(rundir="/mnt/rundir")]

    insert_gcp_secret(container, secret_vol)

    return container


def fv3_container(data_vol: V1Volume, image: str) -> V1Container:
    """A container for running a run-directory from a volume

    Args:
        data_vol: A k8s volume containing a path "rundir" with a saved run-directory.
        image: the prognostic_run docker image

    Returns:
        container for running the output. stores data in the volume at path "rundir".
    """
    runfile_path = "/fv3net/workflows/prognostic_c48_run/sklearn_runfile.py"
    fv3_container = V1Container(name="fv3")
    fv3_container.image = image
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
        V1VolumeMount(mount_path="/mnt", name=data_vol.name),
    ]

    return fv3_container


def post_process_container(
    path: str,
    destination: str,
    data_vol: V1Volume,
    secret_vol: V1Volume,
    image: str = "us.gcr.io/vcm-ml/fv3net:9ed905f7f3ec44484370b8f915834744fe49c636",
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
    container.working_dir = "/home/jovyan/fv3net/workflows/prognostic_c48_run"
    container.command = ["python", "post_process.py", rundir, destination]
    # Suitable for C48 job
    container.resources = V1ResourceRequirements(
        limits=dict(cpu="6", memory="3600M"), requests=dict(cpu="6", memory="3600M"),
    )

    insert_gcp_secret(container, secret_vol)

    return container


def post_processed_fv3_pod_spec(
    model_config: Mapping,
    output_url: str,
    fv3config_image: str,
    fv3_image: str,
    post_process_image: str,
) -> V1PodSpec:
    """A PodSpec for running the prognostic run
    
    Runs FV3 with the fv3config object ``model_config`` and saves the post
    processed output to ``output_url``.

    This orchestrates three containers. The first writes the run-directory,
    the second runs the model, and third post-processes and uploads the data.
    """

    empty_vol = V1Volume(name="rundir")
    empty_vol.empty_dir = V1EmptyDirVolumeSource()

    secret_vol = V1Volume(name="google-secret")
    secret_vol.secret = V1SecretVolumeSource(secret_name="gcp-key")

    # Need to add toleration for large jobs
    climate_sim_toleration = V1Toleration(
        effect="NoSchedule", value="climate-sim-pool", key="dedicated", operator="Equal"
    )

    return V1PodSpec(
        init_containers=[
            write_rundir_container(
                model_config, empty_vol, secret_vol, fv3config_image
            ),
            fv3_container(empty_vol, image=fv3_image),
        ],
        containers=[
            post_process_container(
                "rundir", output_url, empty_vol, secret_vol, image=post_process_image
            )
        ],
        volumes=[empty_vol, secret_vol],
        restart_policy="Never",
        tolerations=[climate_sim_toleration],
    )
