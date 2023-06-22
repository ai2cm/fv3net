import click
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


@click.group()
def cli():
    """
    This command can write a rundir, prepare config, and execute run.
    """
    pass


@cli.command()
@click.argument("config")
@click.argument("rundir")
def write_rundir(config: str, rundir: str):
    """Write a scream run directory given a scream config and a rundir
    For each given processor count, a new directory will be created under
    ${CASE_ROOT}/${CASE_NAME}/${processor_count}x1
    Args:
      config: path to scream config, either local or in the cloud
      rundir: path to rundir
    """
    os.makedirs(rundir, exist_ok=True)
    scream_config = _make_scream_config(config)
    local_output_yaml, local_run_script = get_local_output_yaml_and_run_script(
        scream_config, rundir
    )
    command = scream_config.compose_write_scream_run_directory_command(
        local_output_yaml, local_run_script
    )
    subprocess.run(command, shell=True)


@cli.command()
@click.argument("input_string")
@click.argument("output_config")
@click.argument("precompiled_case")
def prepare_config(input_string: str, output_config: str, precompiled_case: bool):
    """Prepare config for scream run
    Args:
        input_string: either a config file or yaml passed as string
        output_config: file path to the saved output config
        precompiled_case: whether to use a pre-compiled case
    """
    try:
        fs = vcm.cloud.get_fs(input_string)
        if fs.exists(input_string):
            shutil.copy(input_string, output_config)
    except ValueError:
        logger.info("Input is not a config file, writing to output config")
        subprocess.run(
            f"cat << EOF > {output_config}\n{input_string}\nEOF", shell=True,
        )
    if precompiled_case:
        logger.info("Using a pre-compiled case")
        subprocess.run(f"sed -i '/create_newcase/d' {output_config}", shell=True)
        subprocess.run(f'echo "\ncreate_newcase: False" >> {output_config}', shell=True)


@cli.command()
@click.argument("config")
@click.argument("rebuild")
def execute(config: str, rebuild: bool):
    """Execute scream run

    Args:
        config: file path to scream config
        rebuild: whether to rebuild the case
    """
    scream_config = _make_scream_config(config)
    scream_config.submit_scream_run(rebuild)
