import logging
import vcm
import fv3config
import os

from .append import append_segment_to_run_url as append
from .validate import validate_chunks

__all__ = ["append", "create"]

logger = logging.getLogger(__name__)


def create(url: str, fv3config_dict: dict):
    """Initialize a segmented run at ``url`` from a fv3config dictionary"""
    logger.info(f"Setting up segmented run at {url}")
    fs = vcm.cloud.get_fs(url)
    if fs.exists(url):
        raise FileExistsError(
            f"The given url '{url}' contains an object. Delete "
            "everything under output url and resubmit."
        )
    validate_chunks(fv3config_dict)
    fs.makedirs(url, exist_ok=True)
    with fs.open(os.path.join(url, "fv3config.yml"), "w") as f:
        fv3config.dump(fv3config_dict, f)
