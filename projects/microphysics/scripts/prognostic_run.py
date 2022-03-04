#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
import dacite
import warnings

import wandb
import yaml
from runtime.segmented_run import api
from runtime.segmented_run.prepare_config import HighLevelConfig
import emulation

from fv3net.artifacts.resolve_url import resolve_url

from config import BUCKET

logging.basicConfig(level=logging.INFO)


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="",
    help="path to microphysics emulation model...should probably end with .tf",
)
parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="A unique tag. Can be used to look-up these outputs in subsequent timesteps.",
)
parser.add_argument("--segments", "-n", type=int, default=1, help="number of segments")
parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)

# online/offine flag
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--offline", dest="online", action="store_false", help="ML is offline"
)
group.add_argument(
    "--online",
    dest="online",
    action="store_true",
    help="ML is online. The is the default.",
)
parser.set_defaults(online=True)

args = parser.parse_args()

job = wandb.init(
    job_type=os.getenv("WANDB_JOB_TYPE", "prognostic_run"),
    project=os.getenv("WANDB_PROJECT", "microphysics-emulation"),
    entity="ai2cm",
)

if args.tag:
    job.tags = job.tags + (args.tag,)

tag = args.tag or job.id

with args.config_path.open() as f:
    config = yaml.safe_load(f)

config = dacite.from_dict(HighLevelConfig, config)
config.namelist["gfs_physics_nml"]["emulate_zc_microphysics"] = args.online

if args.model:
    if config.zhao_carr_emulation.model is not None:
        warnings.warn(
            UserWarning(
                f"Overiding .zhao_carr_emulation.model.path configuration in yaml "
                "with {args.model}."
            )
        )
    config.zhao_carr_emulation.model = emulation.ModelConfig(path=args.model)

config_dict = config.to_fv3config()

url = resolve_url(BUCKET, job.project, tag)

wandb.config.update(
    {"config": config_dict, "rundir": url, "env": {os.getenv("COMMIT_SHA", "")}}
)

api.create(url, config_dict)
for i in range(args.segments):
    logging.info(f"Running segment {i+1} of {args.segments}")
    api.append(url)

artifact = wandb.Artifact(tag, type="prognostic-run")
artifact.add_reference(url, checksum=False)
wandb.log_artifact(artifact)
