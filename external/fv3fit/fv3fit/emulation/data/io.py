import os
import fsspec
from typing import List, Optional

from vcm import get_fs


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
