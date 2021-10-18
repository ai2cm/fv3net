"""
Implementation Notes:

This program is implemented using an entity-based paradigm. This is suitable
for distributed workflows that modify remote resources.

The functions in an entity-based API only allow for operations create, read,
update, and delete (CRUD). These operations are applied to the entities to
define the behavior of the system. Using only CRUD operations moves the state
of the run execution into these entities, and makes the logic less reliant on
local state, and therefore easier to translate to a distributed context.

For a segmented run "Runs" and "Segments" are the main entities. Runs have
many segments and some global data stores.
"""
import logging
import os
import sys

import click
import fv3config
import fsspec

import vcm

from .run import run_segment
from .append import append_segment_to_run_url
from .validate import validate_chunks

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """
    This script can create segmented runs and appending segments to them.  It
    also provides low level commands used for debugging.  A segmented run can be
    local or on Google Cloud Storage.  Google Cloud Storage locations should
    start with 'gs:'.
    """
    pass


@cli.command()
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


@cli.command()
@click.argument("url")
def append(url: str):
    """Append a segment to a segmented run"""
    append_segment_to_run_url(url)


@cli.command("run-native")
@click.argument("fv3config_path")
@click.argument("rundir")
def run_native(fv3config_path: str, rundir: str):
    """Setup a run-directory and run the model. Used for testing/debugging."""
    with open(fv3config_path) as f:
        config = fv3config.load(f)
    sys.exit(run_segment(config, rundir))
