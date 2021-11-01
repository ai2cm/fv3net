import argparse
import logging
import os
import uuid
from pathlib import Path
from typing import Any
import dacite

import wandb
import yaml
from runtime.segmented_run import api
from runtime.segmented_run.prepare_config import HighLevelConfig

from fv3net.artifacts.resolve_url import resolve_url

logging.basicConfig(level=logging.INFO)

PROJECT = "2021-10-14-microphsyics-emulation-paper"
BUCKET = "vcm-ml-scratch"


def prepare_config(user_config: Any, duration: str, initial_condition: str):
    """Create an fv3config object from a ``user_config``
    
    Args:
        user_config: the input to ``to_fv3config``
        duration: the length of the run
        initial_condition: the path to the initial condition
    """
    return user_config.to_fv3config()


def get_env(args):
    env = {}
    env["TF_MODEL_PATH"] = args.model
    env["OUTPUT_FREQ_SEC"] = str(10800)
    env["SAVE_ZARR"] = "True"
    return env


CONFIG_PATH = Path(__file__).parent / "fv3config.yml"

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=str, default="1d")
parser.add_argument(
    "--model",
    type=str,
    default="gs://vcm-ml-experiments/2021-10-14-microphsyics-emulation-paper/models/all-tends-limited/all-tends-limited-dense/model.tf",  # noqa
    help="path to microphysics emulation model...should probably end with .tf",
)
parser.add_argument(
    "--initial-condition",
    type=str,
    default="gs://vcm-ml-experiments/online-emulator/2021-08-09/gfs-initialized-baseline-06/fv3gfs_run/artifacts/20160601.000000/RESTART",  # noqa
    help="URL to initial conditions (e.g. a restart directory)",
)
parser.add_argument("--tag", type=str, default=uuid.uuid4().hex)
parser.add_argument("--segments", "-n", type=int, default=1, help="number of segments")

args = parser.parse_args()


job = wandb.init(job_type="prognostic_run", project="crapcrap", entity="ai2cm")

with CONFIG_PATH.open() as f:
    config = yaml.safe_load(f)

config = dacite.from_dict(HighLevelConfig, config)
config.initial_conditions = args.initial_condition
config.duration = args.duration

url = resolve_url(BUCKET, PROJECT, args.tag)
env = get_env(args)

wandb.config.update({"config": config, "env": env})
os.environ.update(env)

api.create(url, config)
for i in range(args.segments):
    logging.info(f"Running segment {i+1} of {args.segments}")
    api.append(url)

artifact = wandb.Artifact("prognostic-run", type="prognostic-run")
artifact.add_reference(url)
wandb.log_artifact(artifact)
