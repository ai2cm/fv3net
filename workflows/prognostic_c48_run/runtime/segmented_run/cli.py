import logging
import os

import click
import fv3config
import fsspec

import vcm
from .validate import validate_chunks

logger = logging.getLogger(__name__)


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
    vcm.cloud.copy(fv3config_path, os.path.join(url, "fv3config.yml"))
