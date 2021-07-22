import logging
import os
import shutil

import click
import fv3config
import fsspec

import vcm
from .validate import validate_chunks

logger = logging.getLogger(__name__)


def copy(source: str, destination: str):
    """Copy between any two 'filesystems'. Do not use for large files."""
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)


@click.command()
@click.argument("url")
@click.argument("fv3config_path")
def create(url: str, fv3config_path: str):
    """Initialize segmented run at URL given FV3CONFIG_PATH."""
    logger.info(f"Setting up segmented run at {url}")
    fs = vcm.cloud.get_fs(url)
    if fs.exists(url):
        raise FileExistsError(
            f"The given url '{url}' contains an object. Delete "
            "everything under output url and resubmit."
        )

    with fsspec.open(fv3config_path) as f:
        fv3config_dict = fv3config.load(f)
    validate_chunks(fv3config_dict)
    copy(fv3config_path, os.path.join(url, "fv3config.yml"))
