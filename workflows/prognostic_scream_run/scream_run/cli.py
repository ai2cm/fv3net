import argparse
import fsspec
import os
import shutil
import sys
import yaml
from .config import ScreamConfig
import subprocess
import vcm.cloud
import logging

# TODO: importlib.resources.files is not available prior to python 3.9
if sys.version_info.major == 3 and sys.version_info.minor < 9:
    import importlib_resources  # type: ignore
elif sys.version_info.major == 3 and sys.version_info.minor >= 9:
    import importlib.resources as importlib_resources  # type: ignore

logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--rebuild", help="Whether to rebuild the case", default=True,
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
        local_output_yaml, local_run_script
    )
    subprocess.run(command, shell=True)


def scream_run():
    args = _parse_scream_run_command_args()
    scream_config = _make_scream_config(args.config)
    scream_config.submit_scream_run(args.rebuild)


def _parse_check_config_file_command_args():
    parser = argparse.ArgumentParser("check_config_file_command")
    parser.add_argument(
        "input_string",
        help="A local path or a URI to scream yaml file, \
            or a string of the config itself.",
    )
    parser.add_argument(
        "output_config", help="Output config file name. Must be a local path.",
    )
    parser.add_argument(
        "--precompiled_case", help="Whether it uses a pre-compiled case", default=True,
    )
    return parser.parse_args()


def prepare_scream_config():
    args = _parse_check_config_file_command_args()
    try:
        fs = vcm.cloud.get_fs(args.input_string)
        if fs.exists(args.input_string):
            shutil.copy(args.input_string, args.output_config)
    except ValueError:
        logger.info("Input is not a config file, writing to output config")
        subprocess.run(
            f"cat << EOF > {args.output_config}\n{args.input_string}\nEOF", shell=True,
        )
    if args.precompiled_case:
        logger.info("Using a pre-compiled case")
        subprocess.run(f"sed -i '/create_newcase/d' {args.output_config}", shell=True)
        subprocess.run(
            f'echo "\ncreate_newcase: False" >> {args.output_config}', shell=True
        )
