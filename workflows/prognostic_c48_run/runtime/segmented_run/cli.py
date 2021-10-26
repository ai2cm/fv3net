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
import sys

import click
import fv3config
import fsspec

from .run import run_segment
from . import api

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
    with fsspec.open(fv3config_path) as f:
        fv3config_dict = fv3config.load(f)
    api.create(url, fv3config_dict)


@cli.command()
@click.argument("url")
def append(url: str):
    """Append a segment to a segmented run"""
    sys.exit(api.append(url))


@cli.command("run-native")
@click.argument("fv3config_path")
@click.argument("rundir")
def run_native(fv3config_path: str, rundir: str):
    """Setup a run-directory and run the model. Used for testing/debugging."""
    with open(fv3config_path) as f:
        config = fv3config.load(f)
    sys.exit(run_segment(config, rundir))
