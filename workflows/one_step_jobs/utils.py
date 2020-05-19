import logging
import os
import fsspec
import pprint
from toolz import assoc
import uuid
import yaml
import re
from copy import deepcopy
from typing import List, Dict
from kubernetes.client import V1Job

import fv3config
import fv3kube
from vcm.cloud import get_fs

STDOUT_FILENAME = "stdout.log"
VERTICAL_GRID_FILENAME = "fv_core.res.nc"

SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = SECONDS_IN_MINUTE * 60
SECONDS_IN_DAY = SECONDS_IN_HOUR * 24

KUBERNETES_CONFIG_DEFAULT = {
    "docker_image": "us.gcr.io/vcm-ml/fv3gfs-python",
    "cpu_count": 6,
    "gcp_secret": "gcp-key",
    "image_pull_policy": "Always",
    "capture_output": False,
    "memory_gb": 6.0,
}

KUBERNETES_NAMESPACE = "default"

logger = logging.getLogger(__name__)


def _current_date_from_timestep(timestep: str) -> List[int]:
    """Return timestep in the format required by fv3gfs namelist"""
    year = int(timestep[:4])
    month = int(timestep[4:6])
    day = int(timestep[6:8])
    hour = int(timestep[9:11])
    minute = int(timestep[11:13])
    second = int(timestep[13:15])
    return [year, month, day, hour, minute, second]


def _get_initial_condition_assets(input_url: str, timestep: str) -> List[dict]:
    """
    Get list of assets representing initial conditions for this timestep to pipeline.
    """
    initial_condition_assets = fv3kube.update_tiled_asset_names(
        source_url=input_url,
        source_filename="{timestep}.{category}.tile{tile}.nc",
        target_url="INPUT",
        target_filename="{category}.tile{tile}.nc",
        timestep=timestep,
    )

    return initial_condition_assets


def _get_vertical_grid_asset(config_url: str) -> dict:
    """Get asset corresponding to fv_core.res.nc which describes vertical coordinate"""
    return fv3config.get_asset_dict(
        config_url,
        VERTICAL_GRID_FILENAME,
        target_location="INPUT",
        target_name=VERTICAL_GRID_FILENAME,
    )


def _get_experiment_name(first_tag: str, second_tag: str) -> str:
    """Return unique experiment name with only alphanumeric, hyphen or dot"""
    uuid_short = str(uuid.uuid4())[:8]
    exp_name = f"one-step.{first_tag}.{second_tag}.{uuid_short}"
    return re.sub(r"[^a-zA-Z0-9.\-]", "", exp_name)


def _update_config(
    workflow_name: str,
    base_config_version: str,
    user_model_config: dict,
    input_url: str,
    config_url: str,
    timestep: str,
) -> Dict:
    """
    Update kubernetes and fv3 configurations with user inputs
    to prepare for fv3gfs one-step runs.
    """
    base_model_config = fv3kube.get_base_fv3config(base_config_version)
    model_config = fv3kube.update_nested_dict(base_model_config, user_model_config)

    model_config = fv3config.enable_restart(model_config)
    model_config["experiment_name"] = _get_experiment_name(workflow_name, timestep)
    model_config["initial_conditions"] = _get_initial_condition_assets(
        input_url, timestep
    )
    model_config["initial_conditions"].append(_get_vertical_grid_asset(config_url))
    model_config["namelist"]["coupler_nml"].update(
        {
            "current_date": _current_date_from_timestep(timestep),
            "force_date_from_namelist": True,
        }
    )

    return model_config


def _upload_config_files(
    model_config: dict,
    config_url: str,
    local_vertical_grid_file,
    upload_config_filename="fv3config.yml",
) -> str:
    """
    Upload any files to remote paths necessary for fv3config and the
    fv3gfs one-step runs.
    """
    model_config["diag_table"] = fv3kube.transfer_local_to_remote(
        model_config["diag_table"], config_url
    )

    if local_vertical_grid_file is not None:
        fs = get_fs(config_url)
        vfile_path = os.path.join(config_url, VERTICAL_GRID_FILENAME)
        fs.put(local_vertical_grid_file, vfile_path)

    config_path = os.path.join(config_url, upload_config_filename)
    with fsspec.open(config_path, "w") as config_file:
        config_file.write(yaml.dump(model_config))

    return config_path


def get_run_kubernetes_kwargs(user_kubernetes_config, config_url):

    kubernetes_config = fv3kube.update_nested_dict(
        deepcopy(KUBERNETES_CONFIG_DEFAULT), user_kubernetes_config
    )

    if "runfile" in kubernetes_config:
        runfile_path = kubernetes_config["runfile"]
        kubernetes_config["runfile"] = fv3kube.transfer_local_to_remote(
            runfile_path, config_url
        )

    return kubernetes_config


def _get_job(config_url, tmp_dir, labels, **kwargs) -> V1Job:
    job: V1Job = fv3config.run_kubernetes(
        config_url, tmp_dir, submit=False, **kwargs,
    )

    # increase back off limit
    job.spec.backoff_limit = 3

    # make the name better
    job.metadata.name = None
    job.metadata.generate_name = "one-steps-"

    job.metadata.labels = labels
    job.spec.template.metadata.labels = labels

    return job


def submit_jobs(
    timestep_list: List[str],
    workflow_name: str,
    one_step_config: dict,
    input_url: str,
    output_url: str,
    config_url: str,
    base_config_version: str,
    job_labels=None,
    local_vertical_grid_file=None,
) -> None:
    """Submit one-step job for all timesteps in timestep_list"""

    # load API objects needed to submit jobs

    client = fv3kube.initialize_batch_client()

    zarr_url = os.path.join(output_url, "big.zarr")

    logger.info("Working on one-step jobs with arguments:")
    logger.info(pprint.pformat(locals()))
    # kube kwargs are shared by all jobs
    kube_kwargs = get_run_kubernetes_kwargs(one_step_config["kubernetes"], config_url)
    logger.info(
        "To view job statuses: "
        f"`kubectl get jobs -lorchestrator-jobs={job_labels['orchestrator-jobs']}`"
    )
    logger.info(
        "To clean up jobs: "
        f"`kubectl delete jobs -lorchestrator-jobs={job_labels['orchestrator-jobs']}`"
    )

    def config_factory(**kwargs):
        timestep = timestep_list[kwargs["index"]]
        curr_input_url = os.path.join(input_url, timestep)
        curr_config_url = os.path.join(config_url, timestep)

        config = deepcopy(one_step_config)
        kwargs["url"] = zarr_url
        config["fv3config"]["one_step"] = kwargs

        model_config = _update_config(
            workflow_name,
            base_config_version,
            config["fv3config"],
            curr_input_url,
            curr_config_url,
            timestep,
        )
        return _upload_config_files(
            model_config, curr_config_url, local_vertical_grid_file
        )

    def run_job(wait=False, **kwargs):
        """Run a run_kubernetes job

        kwargs are passed workflows/one_step_jobs/runfile.py:post_process

        """
        uid = str(uuid.uuid4())
        labels = assoc(job_labels, "jobid", uid)
        model_config_url = config_factory(**kwargs)

        # the one step workflow doesn't need to upload its run directories any longer
        # since all the data is in the big zarr. Setting outdir to a pod-local path
        # avoids this unecessary upload step.
        local_tmp_dir = "/tmp/null"

        job = _get_job(model_config_url, local_tmp_dir, labels=labels, **kube_kwargs)

        # submit the k8s job
        client.create_namespaced_job(KUBERNETES_NAMESPACE, job)

        if wait:
            fv3kube.wait_for_complete(labels, sleep_interval=10)

    for k, timestep in enumerate(timestep_list):
        if k == 0:
            logger.info("Running the first time step to initialize the zarr store")
            run_job(index=k, init=True, wait=True, timesteps=timestep_list)
        else:
            logger.info(f"Submitting job for timestep {timestep}")
            run_job(index=k, init=False)

    fv3kube.wait_for_complete(job_labels, raise_on_fail=False)
    fv3kube.delete_completed_jobs(job_labels)
