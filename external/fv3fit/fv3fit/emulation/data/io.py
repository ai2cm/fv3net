import hashlib
import os
from typing import List, Optional

import fsspec
import tensorflow as tf
import logging
from vcm import get_fs

CACHE_DIR = ".data"

logger = logging.getLogger(__name__)


def download_cached(path):
    hash = hashlib.md5(path.encode()).hexdigest()
    filepath = os.path.join(CACHE_DIR, hash)
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(filepath):
        logger.info(f"Downloading {path} to {filepath}")
        tf.io.gfile.copy(path, filepath, overwrite=True)
    else:
        logger.debug(f"{path} found in cache at {filepath}.")
    return filepath


def get_nc_files(
    path: str, fs: Optional[fsspec.AbstractFileSystem] = None
) -> List[str]:
    """
    Get a list of netCDF files from a remote/local directory

    Args:
        path: Local or remote gcs path to netCDF directory
        fs: Filesystem object to use for the glob operation
            searching for netCDFs in the path
    """

    if fs is None:
        fs = get_fs(path)

    files = list(fs.glob(os.path.join(path, "*.nc")))

    # we want to preserve information about the remote protocol
    # so any downstream operations can glean that info from the paths
    if "gs" in fs.protocol:
        files = ["gs://" + f for f in files]

    return files
