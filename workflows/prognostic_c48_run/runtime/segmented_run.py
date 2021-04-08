import logging
import os

import click
import fsspec
import fv3config

import vcm
from .validate import validate_config

logger = logging.getLogger(__name__)


@click.command()
@click.argument("url")
@click.argument("fv3config_path")
@click.argument("runfile_path")
def create_run(url: str, fv3config_path: str, runfile_path: str):
    """Initialize segmented run at URL given FV3CONFIG_PATH and RUNFILE_PATH."""
    logger.info(f"Setting up segmented run at {url}")
    fs = vcm.cloud.get_fs(url)
    if fs.exists(url):
        raise FileExistsError(
            f"The given url '{url}' contains an object. Delete "
            "everything under output url and resubmit."
        )

    with fs.open(fv3config_path) as f:
        fv3config_dict = fv3config.load(f)
    validate_config(fv3config_dict)

    fs.put(fv3config_path, os.path.join(url, "fv3config.yml"))
    fs.put(runfile_path, os.path.join(url, "runfile.py"))
