import signal
import time
import sherpa
import sys
import os
import tempfile
import subprocess
import kubernetes
import kubernetes_utils as fv3kube
import config_ops
import uuid


SHERPA_SCRATCH_DIR = "gs://vcm-ml-scratch/sherpa"


# if you use many parameters, you may need to decrease this to prevent experiment
# name from getting too long
PARAMETER_CHARS_IN_NAME = 20

# --- to configure different parameters, edit only these two routines


def preprocess_config(config, parameters):
    experiment_name = get_experiment_name(parameters)
    config["end_to_end"]["experiment"]["name"] = experiment_name
    config["job"]["metadata"]["labels"]["name"] = experiment_name
    config["job"]["metadata"]["name"] = experiment_name
    config["job"]["spec"]["template"]["spec"]["containers"][0]["args"] = (
        config["job"]["spec"]["template"]["spec"]["containers"][0]["args"]).format(
            training_output_dir=os.path.join(SHERPA_SCRATCH_DIR, experiment_name)
    )
    update_config_from_parameters(config, parameters)


def update_config_from_parameters(config, parameters):
    parameters["training_timesteps"]
    parameters["timesteps_per_batch"]
    config["train_sklearn_model"]["files_per_batch"] = parameters["timesteps_per_batch"]
    config["train_sklearn_model"]["num_batches"] = int(
        parameters["training_timesteps"] / parameters["timesteps_per_batch"]
    )


# --- Code below here does not need to be edited to set up new parameters

BASE_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "base_trial_config"
)
LOOP_DELAY = 10
JOB_LIST = []

UID = str(uuid.uuid4())


def get_experiment_name(parameters):
    base_name = "-".join(
        [
            f"{name[:PARAMETER_CHARS_IN_NAME]}-{value}"
            for name, value in sorted(parameters.items(), key=lambda x: x[0])
        ]
    )
    jobname = os.environ.get("JOBNAME")
    if jobname is not None:
        name = "-".join([base_name, jobname, UID[:8]])
    else:
        name = "-".join([base_name, UID[:8]])
    name = name.replace('_', '-')  # underscores not allowed in kube job names
    return name


def get_objective(experiment_name):
    # outdir = os.path.join("gs://vcm-ml-scratch/sherpa/", experiment_name)
    return 1


def get_batch_client() -> kubernetes.client.BatchV1Api:
    try:
        kubernetes.config.load_kube_config()
    except (TypeError, kubernetes.config.config_exception.ConfigException):
        kubernetes.config.load_incluster_config()
    batch_client = kubernetes.client.BatchV1Api()
    return batch_client


BATCH_CLIENT = get_batch_client()


def sigterm_handler(_signo, _stack_frame):
    cancel_jobs()
    sys.exit(0)


def cancel_jobs():
    for job in JOB_LIST:
        BATCH_CLIENT.delete_namespaced_job(
            job.metadata.name, namespace=job.metadata.namespace
        )
        pass


def get_job(experiment_name):
    labels = {"app": "sherpa", "name": experiment_name}
    job_list = fv3kube.list_jobs(BATCH_CLIENT, labels)
    if len(job_list) > 1:
        cancel_jobs()
        raise RuntimeError(f"unexpectedly got {len(job_list)} jobs for labels {labels}")
    else:
        return job_list[0]


if __name__ == "__main__":
    # must terminate remote jobs if Sherpa terminates the trial
    signal.signal(signal.SIGTERM, sigterm_handler)

    client = sherpa.Client(host="localhost", port=27001)

    trial = client.get_trial()
    config = config_ops.load_config(BASE_CONFIG_DIR)
    preprocess_config(config, trial.parameters)
    experiment_name = get_experiment_name(trial.parameters)
    with tempfile.TemporaryDirectory() as tmpdir:
        config_ops.write_config(config, tmpdir)
        subprocess.check_call(["kubectl", "apply", "-k", tmpdir])
        job = get_job(experiment_name)
        JOB_LIST.append(job)
        while True:
            if fv3kube.job_complete(job):
                objective = get_objective(experiment_name)
                client.send_metrics(trial, iteration=1, objective=objective)
                sys.exit(0)
            elif fv3kube.job_failed(job):
                sys.exit(1)
            time.sleep(LOOP_DELAY)
            job = get_job(experiment_name)
