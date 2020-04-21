import logging
import os
import time
import kubernetes
from kubernetes.client import BatchV1Api
import fsspec
import yaml
from typing import Sequence, Mapping, Tuple
from pathlib import Path

import fv3config

from vcm.cubedsphere.constants import RESTART_CATEGORIES, TILE_COORDS_FILENAMES
from vcm.cloud.fsspec import get_protocol, get_fs

logger = logging.getLogger(__name__)

JobInfo = Tuple[str, str]


# Map for configuration defaults required for different fv3gfs-python versions
PWD = Path(os.path.abspath(__file__)).parent
FV3CONFIG_DEFAULTS_BY_VERSION = {
    "v0.2": os.path.join(PWD, "default_yamls/v0.2/fv3config.yml"),
    "v0.3": os.path.join(PWD, "default_yamls/v0.3/fv3config.yml"),
}


def update_nested_dict(source_dict: Mapping, update_dict: Mapping) -> Mapping:
    """
    Recursively update a dictionary with new values.  Used to update
    configuration dicts with partial specifications.
    """
    for key in update_dict:
        if key in source_dict and isinstance(source_dict[key], Mapping):
            update_nested_dict(source_dict[key], update_dict[key])
        else:
            source_dict[key] = update_dict[key]
    return source_dict


def get_base_fv3config(version_key: str) -> Mapping:
    """
    Get base configuration dictionary specific to an fv3gfs-python version.
    """
    config_path = FV3CONFIG_DEFAULTS_BY_VERSION[version_key]
    with fsspec.open(config_path) as f:
        base_yaml = yaml.safe_load(f)

    return base_yaml


def transfer_local_to_remote(path: str, remote_url: str) -> str:
    """
    Transfer a local file to a remote path and return that remote path.
    If path is already remote, this does nothing.
    """
    if get_protocol(path) == "file":
        remote_path = os.path.join(remote_url, os.path.basename(path))
        get_fs(remote_url).put(path, remote_path)
        path = remote_path
    return path


def update_tiled_asset_names(
    source_url: str,
    source_filename: str,
    target_url: str,
    target_filename: str,
    **kwargs,
) -> Sequence[dict]:

    """
    Update tile-based fv3config assets with new names.  Uses format to update
    any data in filename strings with category, tile, and provided keyword
    arguments.

    Filename strings should include any specified variable name inserts to
    be updated with a format. E.g., "{timestep}.{category}.tile{tile}.nc"
    """
    assets = [
        fv3config.get_asset_dict(
            source_url,
            source_filename.format(category=category, tile=tile, **kwargs),
            target_location=target_url,
            target_name=target_filename.format(category=category, tile=tile, **kwargs),
        )
        for category in RESTART_CATEGORIES
        for tile in TILE_COORDS_FILENAMES
    ]

    return assets


def initialize_batch_client() -> kubernetes.client.BatchV1Api:

    try:
        kubernetes.config.load_kube_config()
    except TypeError:
        kubernetes.config.load_incluster_config()

    batch_client = kubernetes.client.BatchV1Api()

    return batch_client


def wait_for_complete(
    job_labels: Mapping[str, str],
    batch_client: kubernetes.client.BatchV1Api = None,
    sleep_interval: int = 30,
    raise_on_fail: bool = True,
):
    """
    Function to block operation until a group of submitted kubernetes jobs complete.

    Args:
        job_labels: key-value pairs of job labels used to filter the job query. These
            values are translated to equivalence statements for the selector. E.g.,
            "key1=value1,key2=value2,..."
        client: A BatchV1Api client to communicate with the cluster.
        sleep_interval: time interval between active job check queries
        raise_on_bool: flag for whether to raise error if any job has failed

    Raises:
        ValueError if any of the jobs fail, when the
        failed jobs are first detected (if raise_on_fail is True)


    """

    if batch_client is None:
        batch_client = initialize_batch_client()

    # Check active jobs
    while True:
        time.sleep(sleep_interval)
        jobs = list_jobs(batch_client, job_labels)
        job_done = _handle_jobs(jobs, raise_on_fail)

        if job_done:
            break

    logger.info("All kubernetes jobs succesfully complete!")


def _handle_jobs(jobs: Sequence, raise_on_fail: bool) -> bool:

    failed = {}
    complete = {}
    active = {}

    for job in jobs:
        name = job.metadata.name
        if job_failed(job):
            failed[name] = job
        elif job_complete(job):
            complete[name] = job
        else:
            active[name] = job

    if len(failed) > 0:
        failed_job_names = list(failed)
        if raise_on_fail:
            raise ValueError(f"These jobs have failed: {failed_job_names}")
        else:
            logger.warning(f"These jobs have failed: {failed_job_names}")

    if len(active) == 0:
        return True
    else:
        log_active_jobs(active)
        return False


def log_active_jobs(active):
    if len(active) > 0:
        jobs = list(active.keys())
        logger.info(
            f"Active Jobs at {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "\n".join(jobs)
        )


def list_jobs(client, job_labels):
    selector_format_labels = [f"{key}={value}" for key, value in job_labels.items()]
    combined_selectors = ",".join(selector_format_labels)
    jobs = client.list_job_for_all_namespaces(label_selector=combined_selectors)

    return jobs.items


def job_failed(job):
    conds = job.status.conditions
    conditions = [] if conds is None else conds
    for cond in conditions:
        if cond.status == "True":
            return cond.type == "Failed"
    return False


def job_complete(job):
    conds = job.status.conditions
    conditions = [] if conds is None else conds
    for cond in conditions:
        if cond.status == "True":
            return cond.type == "Complete"
    return False


def delete_completed_jobs(job_labels: Mapping[str, str], client: BatchV1Api = None):
    client = initialize_batch_client() if client is None else client

    logger.info("Deleting succesful jobs.")
    jobs = list_jobs(client, job_labels)
    for job in jobs:
        if job_complete(job):
            name = job.metadata.name
            logger.info(f"Deleting completed job: {name}")
            client.delete_namespaced_job(name, namespace=job.metadata.namespace)
