import signal
import time
import sherpa
import sys
import os
import yaml
import tempfile
import subprocess
import kubernetes
import fv3kube
import uuid

BASE_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "base_trial_config"
)
LOOP_DELAY = 10
JOB_LIST = []

UID = str(uuid.uuid4())


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


def load_config(config_dir):
    config = {}
    for filename in os.listdir(config_dir):
        if filename[-4:] == ".yml":
            with open(os.path.join(config_dir, filename), "r") as f:
                config[filename[:-4]] = ConfigFile(filename, yaml.load(f))
        elif filename[-5:] == ".yaml":
            with open(os.path.join(config_dir, filename), "r") as f:
                config[filename[:-5]] = ConfigFile(filename, yaml.load(f))
    return config


class ConfigFile:
    def __init__(self, filename, config):
        self.filename = filename
        self.config = config

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value


def preprocess_config(config, parameters):
    experiment_name = get_experiment_name(parameters)
    config["end_to_end"]["experiment"]["name"] = experiment_name
    config["job"]["metadata"]["labels"]["name"] = experiment_name
    config["job"]["metadata"]["name"] = experiment_name
    update_config_from_parameters(config, parameters)


def write_config(config, config_dir):
    for name, config_file in config.items():
        filename = os.path.join(config_dir, config_file.filename)
        with open(filename, "w") as f:
            yaml.dump(config_file.config, f)


def update_config_from_parameters(config, parameters):
    parameters["training_timesteps"]
    parameters["timesteps_per_batch"]
    config["train_sklearn_model"]["files_per_batch"] = parameters["timesteps_per_batch"]
    config["train_sklearn_model"]["num_batches"] = int(
        parameters["training_timesteps"] / parameters["timesteps_per_batch"]
    )


def get_experiment_name(parameters):
    base_name = "-".join(
        [
            f"{name}={value}"
            for name, value in sorted(parameters.items(), key=lambda x: x[0])
        ]
    )
    jobname = os.environ.get("JOBNAME")
    if jobname is not None:
        return "-".join([base_name, jobname, UID[:8]])
    else:
        return "-".join([base_name, UID[:8]])


def list_jobs(client, job_labels):
    selector_format_labels = [f"{key}={value}" for key, value in job_labels.items()]
    combined_selectors = ",".join(selector_format_labels)
    jobs = client.list_job_for_all_namespaces(label_selector=combined_selectors)
    return jobs.items


def get_job(experiment_name):
    labels = {"app": "sherpa", "name": experiment_name}
    job_list = fv3kube.list_jobs(BATCH_CLIENT, labels)
    if len(job_list) > 0:
        cancel_jobs()
        raise RuntimeError(f"unexpectedly got {len(job_list)} jobs for labels {labels}")


def get_objective(experiment_name):
    outdir = os.path.join("gs://vcm-ml-scratch/sherpa/", experiment_name)


if __name__ == "__main__":
    # must terminate remote jobs if Sherpa terminates the trial
    signal.signal(signal.SIGTERM, sigterm_handler)

    client = sherpa.Client(host="localhost", port=27001)

    trial = client.get_trial()
    config = load_config(BASE_CONFIG_DIR)
    preprocess_config(config, trial.parameters)
    experiment_name = config["job"]["metadata"]["labels"]["name"]
    with tempfile.TemporaryDirectory() as tmpdir:
        write_config(config, tmpdir)
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
