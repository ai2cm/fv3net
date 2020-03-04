import logging

import xarray as xr
from dask.delayed import delayed

from vcm.cloud.fsspec import get_fs
from vcm.convenience import open_delayed

__all__ = ["open_tiles"]


def _read_metadata_remote(fs, url):
    logging.info("Reading metadata")
    with fs.open(url, "rb") as f:
        return xr.open_dataset(f)


def _open_remote_nc(fs, url):
    with fs.open(url, "rb") as f:
        return xr.open_dataset(f).load()


def open_tiles(url_prefix: str) -> xr.Dataset:
    """Lazily open a set of FV3 netCDF tile files

    Args:
        url_prefix: the prefix of the set of files before ".tile?.nc". The metadata
            and dimensions are harvested from the first tile only.
    Returns:
        dataset of merged tiles.

    """
    fs = get_fs(url_prefix)
    files = sorted(fs.glob(url_prefix + ".tile?.nc"))
    schema = _read_metadata_remote(fs, files[0])
    delayeds = [delayed(_open_remote_nc)(fs, url) for url in files]
    datasets = [open_delayed(d, schema) for d in delayeds]
    return xr.concat(datasets, dim="tile")
