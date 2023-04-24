import os
import sys

# TODO: importlib.resources.files is not available prior to python 3.9
if sys.version_info.major == 3 and sys.version_info.minor < 9:
    import importlib_resources  # type: ignore
elif sys.version_info.major == 3 and sys.version_info.minor >= 9:
    import importlib.resources as importlib_resources  # type: ignore
import subprocess
from .config import ScreamConfig
from dataclasses import asdict


def compose_run_scream_commands(config: ScreamConfig, target_directory: str):
    os.makedirs(os.path.dirname(target_directory), exist_ok=True)
    command = ""
    for key, value in asdict(config).items():
        if isinstance(value, list):
            value = ",".join(value)
        command += f" --{key} {value}"
    return command


def execute_run(command, target_directory: str):
    run_script = importlib_resources.files("scream_run.template").joinpath(
        "run_eamxx.sh"
    )
    local_script = os.path.join(target_directory, os.path.basename(run_script))
    subprocess.check_call(["cp", str(run_script), local_script])
    command = local_script + command
    subprocess.run(command, shell=True)
