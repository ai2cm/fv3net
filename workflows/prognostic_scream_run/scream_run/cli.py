import argparse
import fsspec
import os
import shutil
import sys
import yaml
from .config import ScreamConfig
import subprocess

# TODO: importlib.resources.files is not available prior to python 3.9
if sys.version_info.major == 3 and sys.version_info.minor < 9:
    import importlib_resources  # type: ignore
elif sys.version_info.major == 3 and sys.version_info.minor >= 9:
    import importlib.resources as importlib_resources  # type: ignore


def _parse_write_scream_run_directory_command_args():
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


def get_local_output_yaml_and_run_script(scream_config: ScreamConfig, rundir: str):
    run_script = importlib_resources.files("scream_run.template").joinpath(
        "run_eamxx.sh"
    )
    local_run_script = os.path.join(rundir, os.path.basename(run_script))
    shutil.copy(run_script, local_run_script)
    local_output_yaml = scream_config.get_local_output_yaml(rundir)
    return local_output_yaml, local_run_script


def write_scream_run_directory():
    args = _parse_write_scream_run_directory_command_args()
    os.makedirs(args.rundir, exist_ok=True)
    scream_config = _make_scream_config(args.config)
    local_output_yaml, local_run_script = get_local_output_yaml_and_run_script(
        scream_config, args.rundir
    )
    command = scream_config.compose_write_scream_run_directory_command(
        args.rundir, local_output_yaml, local_run_script
    )
    subprocess.run(command, shell=True)


# TODO: add tests for scream run as part of the docker image PR
def scream_run():
    args = _parse_scream_run_command_args()
    scream_config = _make_scream_config(args.config)
    scream_config.submit_scream_run()
