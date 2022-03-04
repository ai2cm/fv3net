#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
import dacite

import wandb
import yaml
from runtime.segmented_run import api
from runtime.segmented_run.prepare_config import HighLevelConfig

from fv3net.artifacts.resolve_url import resolve_url

from config import BUCKET

logging.basicConfig(level=logging.INFO)


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def run_from_config_dict(config: dict):
    # pop orchestration related args
    config = {**config}
    tag = config.pop("tag")
    segments = config.pop("segments", 1)

    url = resolve_url(BUCKET, job.project, tag)
    wandb.config.update(
        {
            "config": config,
            "rundir": url,
            "env": {"COMMIT_SHA": os.getenv("COMMIT_SHA", "")},
        }
    )
    config = dacite.from_dict(HighLevelConfig, config)
    config_dict = config.to_fv3config()

    api.create(url, config_dict)
    for i in range(segments):
        logging.info(f"Running segment {i+1} of {segments}")
        api.append(url)

    artifact = wandb.Artifact(tag, type="prognostic-run")
    artifact.add_reference(url, checksum=False)
    wandb.log_artifact(artifact)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="A unique tag. Can be used to look-up these outputs in subsequent timesteps.",
)
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


with args.config_path.open() as f:
    config = yaml.safe_load(f)

config["namelist"]["gfs_physics_nml"]["emulate_zc_microphysics"] = args.online
config["tag"] = args.tag or job.id
run_from_config_dict(config)
