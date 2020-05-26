import logging
import os
import fsspec
import pprint
from toolz import assoc
import uuid
import yaml
from copy import deepcopy
from typing import List
from kubernetes.client import V1Job

import fv3config
import fv3kube
import vcm

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


def _upload_config_files(
    model_config: dict, config_url: str, upload_config_filename="fv3config.yml",
) -> str:
    """
    Upload any files to remote paths necessary for fv3config and the
    fv3gfs one-step runs.
    """
    model_config["diag_table"] = fv3kube.transfer_local_to_remote(
        model_config["diag_table"], config_url
    )

    config_path = os.path.join(config_url, upload_config_filename)
    with fsspec.open(config_path, "w") as config_file:
        config_file.write(yaml.dump(model_config))

    return config_path


def get_run_kubernetes_kwargs(user_kubernetes_config, config_url):

    kubernetes_config = vcm.update_nested_dict(
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
        curr_config_url = os.path.join(config_url, timestep)

        config = deepcopy(one_step_config)
        kwargs["url"] = zarr_url
        config["one_step"] = kwargs

        model_config = fv3kube.get_full_config(config, input_url, timestep,)
        return _upload_config_files(model_config, curr_config_url)

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
