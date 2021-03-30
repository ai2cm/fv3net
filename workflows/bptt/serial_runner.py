import argparse
import yaml
import os
import sherpa
import subprocess
import re
import sys
from collections import namedtuple

TRAIN_CONFIG_FILENAME = "training.yaml"

MAX_PARAMETER_CHARS_IN_NAME = 16

VAL_LOSS_PATTERN = re.compile(r"val_loss: ([0-9]+\.[0-9]+)")

TrainArgs = namedtuple(
    "TrainArgs",
    "train_data_path train_config_file output_data_path timesteps_file "
    "validation_timesteps_file",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", help="configuration yaml location", type=str)
    parser.add_argument("datadir", help="local dataset location", type=str)
    parser.add_argument("outdir", help="output location for training trials", type=str)
    return parser.parse_args()


# --- to configure different parameters, edit only this routine
def update_config_from_parameters(config, parameters):
    """Takes in a nested dictionary representing the trial configuration directory,
    and updates it in-place using the trial hyperparameters.

    The keys of config are filename roots from base_trial_config, and the values
    represent the corresponding yaml file. A yaml file with multiple documents
    (separated by dashed lines ----) is represented by a list of dictionaries, while
    a single-document yaml file is just a dictionary.
    """
    config["training"]["optimizer"]["kwargs"]["learning_rate"] = float(
        parameters.get("learning_rate", 1e-3)
    )
    config["training"]["hyperparameters"]["n_units"] = int(
        parameters.get("n_units", 256)
    )
    config["training"]["hyperparameters"]["n_hidden_layers"] = int(
        parameters.get("n_hidden_layers", 3)
    )
    config["training"]["hyperparameters"]["state_noise"] = float(
        parameters.get("state_noise", 0.0)
    )
    config["training"]["hyperparameters"]["use_moisture_limiter"] = bool(
        parameters.get("use_q_limiter", False)
    )
    config["training"]["regularizer"]["kwargs"]["l"] = float(parameters.get("l2", 1e-4))
    config["training"]["decreased_learning_rate"] = (
        float(parameters.get("lr_reduction", 1e-1))
        * config["training"]["optimizer"]["kwargs"]["learning_rate"]
    )
    config["training"]["random_seed"] = int(parameters.get("random_seed", 0))


CWD = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG_DIR = os.path.join(CWD, "base_trial_config")
OUTPUT_DIR = os.path.join(CWD, "output")


def preprocess_config(config, parameters):
    update_config_from_parameters(config, parameters)


def get_experiment_name(parameters):
    name = "-".join(
        [
            f"{name[:MAX_PARAMETER_CHARS_IN_NAME]}-{value:.03g}"
            for name, value in sorted(parameters.items(), key=lambda x: x[0])
        ]
    )
    name = name.replace("_", "-")  # underscores not allowed in kube job names
    return name


def load_config_dir(dirname):
    config = {}
    for filename in os.listdir(dirname):
        if filename[-5:] == ".yaml":
            base_name = filename[:-5]
            with open(os.path.join(dirname, filename)) as f:
                config[base_name] = yaml.safe_load(f)
    return config


def write_config_dir(config, config_dir):
    for name, config_file in config.items():
        filename = os.path.join(config_dir, name + ".yaml")
        with open(filename, "w") as f:
            yaml.safe_dump(config_file, f)


if __name__ == "__main__":
    args = parse_args()
    with open(args.configfile, "r") as f:
        config = yaml.safe_load(f)
    algorithm_name = config["algorithm"]["name"]
    if not hasattr(sherpa.algorithms, algorithm_name):
        raise ValueError(
            f"No sherpa algorithm {algorithm_name} exists, is there a typo?"
        )
    algorithm = getattr(sherpa.algorithms, algorithm_name)(
        *config["algorithm"].get("args", []), **config["algorithm"].get("kwargs", {})
    )
    parameters = []
    for parameter_config in config["parameters"]:
        parameters.append(sherpa.core.Parameter.from_dict(parameter_config))
    study_output_dir = os.path.join(args.outdir, "study")
    os.makedirs(study_output_dir, exist_ok=True)
    study = sherpa.Study(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=True,
        output_dir=study_output_dir,
        dashboard_port=8880,
    )
    for trial in study:
        print(f"Running trial {trial.parameters}")
        experiment_name = get_experiment_name(trial.parameters)

        config = load_config_dir(BASE_CONFIG_DIR)
        update_config_from_parameters(config, trial.parameters)
        outdir = os.path.join(os.path.abspath(args.outdir), experiment_name)
        os.makedirs(outdir, exist_ok=True)
        write_config_dir(config, outdir)
        lines = []
        command = [
            "python3",
            "train.py",
            args.datadir,
            os.path.join(outdir, TRAIN_CONFIG_FILENAME),
            outdir,
        ]
        print(" ".join(command))
        # train_args = TrainArgs(
        #     DATA_PATH, os.path.join(outdir, TRAIN_CONFIG_FILENAME), outdir, None, None
        # )
        # old_stdout = sys.stdout
        # result = StringIO()
        # fv3fit_main(train_args)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, encoding="utf-8",)
        i_observation = 0
        for line in iter(process.stdout.readline, ""):
            match = re.search(VAL_LOSS_PATTERN, line)
            if match:
                sys.stdout.write(line)
                loss = float(match.group(1))
                study.add_observation(trial, loss, iteration=i_observation)
                i_observation += 1
        if process.returncode not in (0, None):
            # raise subprocess.SubprocessError(
            print(f"Process exited with non-zero return code {process.returncode}")
        study.finalize(trial)
        study.save()
    print(f"Best results: {study.get_best_result()}")
