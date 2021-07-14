import os
from typing import List

from vcm import get_fs


def get_nc_files(path: str) -> List[str]:
    """
    Get a list of netCDF files from a remote/local directory

    Args:
        path: Local or remote gcs path to netCDF directory
    """

    fs = get_fs(path)
    files = list(fs.glob(os.path.join(path, "*.nc")))

    return files
