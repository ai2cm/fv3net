import argparse
import fsspec
import os
import yaml
from .config import ScreamConfig
import subprocess


def _parse_run_scream_command_args():
    parser = argparse.ArgumentParser("run_scream_command")
    parser.add_argument(
        "config", help="URI to scream yaml file. Supports any path used by fsspec."
    )
    parser.add_argument(
        "rundir", help="Desired output directory. Must be a local directory"
    )
    return parser.parse_args()


def _make_scream_config(config: str, rundir: str):
    with fsspec.open(config) as f:
        scream_config = ScreamConfig.from_dict(yaml.safe_load(f))
        scream_config.resolve_output_yaml(rundir)
    return scream_config


def scream_run():
    args = _parse_run_scream_command_args()
    os.makedirs(args.rundir, exist_ok=True)
    scream_config = _make_scream_config(args.config, args.rundir)
    command = scream_config.compose_run_scream_commands(args.rundir)
    subprocess.run(command, shell=True)
