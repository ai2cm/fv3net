import argparse
from .run_scream import compose_run_scream_commands, execute_run
import fsspec
import yaml
from .config import ScreamConfig


def _parse_run_scream_command_args():
    parser = argparse.ArgumentParser("run_scream_command")
    parser.add_argument(
        "config", help="URI to scream yaml file. Supports any path used by fsspec."
    )
    parser.add_argument(
        "rundir", help="Desired output directory. Must be a local directory"
    )
    return parser.parse_args()


def _make_scream_config(config: str, output_yaml_path: str):
    with fsspec.open(config) as f:
        scream_config = ScreamConfig.from_dict(yaml.safe_load(f))
        scream_config.resolve_output_yaml(output_yaml_path)
    return scream_config


def scream_run():
    args = _parse_run_scream_command_args()
    scream_config = _make_scream_config(args.config, args.rundir)
    command = compose_run_scream_commands(scream_config, args.rundir)
    execute_run(command, args.rundir)
