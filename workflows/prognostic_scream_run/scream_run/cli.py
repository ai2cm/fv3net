import argparse
import fsspec
import os
import yaml
from .config import ScreamConfig
import subprocess


def _parse_write_scream_run_command_args():
    parser = argparse.ArgumentParser("write_scream_run_directory_command")
    parser.add_argument(
        "config", help="URI to scream yaml file. Supports any path used by fsspec."
    )
    parser.add_argument(
        "rundir", help="Desired output directory. Must be a local directory"
    )
    return parser.parse_args()


def _parse_scream_run_command_args():
    parser = argparse.ArgumentParser("scream_run_command")
    parser.add_argument(
        "config", help="URI to scream yaml file. Supports any path used by fsspec."
    )
    return parser.parse_args()


def _make_scream_config(config: str):
    with fsspec.open(config) as f:
        scream_config = ScreamConfig.from_dict(yaml.safe_load(f))
    return scream_config


def write_scream_run_directory():
    args = _parse_write_scream_run_command_args()
    os.makedirs(args.rundir, exist_ok=True)
    scream_config = _make_scream_config(args.config)
    command = scream_config.compose_write_scream_run_directory_command(args.rundir)
    subprocess.run(command, shell=True)


# TODO: add tests for scream run as part of the docker image PR
def scream_run():
    args = _parse_scream_run_command_args()
    scream_config = _make_scream_config(args.config)
    scream_config.submit_scream_run()
