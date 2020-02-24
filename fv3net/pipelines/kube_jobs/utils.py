import logging
import os
import time
import kubernetes
from typing import List, Mapping

import fv3config

from vcm.cubedsphere.constants import RESTART_CATEGORIES, TILE_COORDS_FILENAMES
from vcm.cloud.fsspec import get_protocol, get_fs

logger = logging.getLogger(__name__)


def update_nested_dict(source_dict: dict, update_dict: dict) -> dict:
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
) -> List[dict]:

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


def wait_for_complete(job_labels: Mapping[str, str]):
    """
    Function to block operation until a group of submitted kubernetes jobs complete.

    Args:
        job_labels: key-value pairs of job labels used to filter the job query. These
            values are translated to equivalence statements for the selector. E.g.,
            "key1=value1,key2=value2,..."
    """

    selector_format_labels = [f"{key}={value}" for key, value in job_labels.items()]
    combined_selectors = ",".join(selector_format_labels)

    kubernetes.config.load_kube_config()
    batch = kubernetes.client.BatchV1Api()

    # Check active jobs
    while True:

        jobs = batch.list_job_for_all_namespaces(label_selector=combined_selectors)
        active_jobs = [
            job_info.metadata.name for job_info in jobs.items if job_info.status.active
        ]

        if active_jobs:
            logger.info(
                f"Active Jobs at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                + "\n".join(active_jobs)
            )
            time.sleep(30)
            continue
        else:
            logger.info("All kubernetes jobs finished!")
            break

    # Get failed and successful jobs
    failed = []
    success = []
    for job_info in jobs.items:
        if job_info.status.failed:
            failed.append(job_info.metadata.name)
        elif job_info.status.succeeded:
            # tuples for delete operation
            success.append((job_info.metadata.name, job_info.metadata.namespace))

    logger.info("Failed Jobs from EXP_LABEL:\n" + "\n".join(failed))

    # Delete successful jobs
    logger.info("Deleting successful jobs.")
    for delete_args in success:
        jobname, namespace = delete_args
        logger.info(f"Deleting -- {jobname}")
        batch.delete_namespaced_job(jobname, namespace)
