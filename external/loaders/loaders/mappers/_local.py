import logging as logger
import os
from functools import partial
from multiprocessing import Pool

import fsspec
import xarray as xr

from ._base import GeoMapper


class LocalMapper(GeoMapper):
    """A mapper for a directory of netCDFs. One per time-step.
    """

    def __init__(self, path: str):
        self.fs = fsspec.get_fs_token_paths(path)[0]
        self.path = path

    def _files(self):
        files = self.fs.glob(os.path.join(self.path, "*.nc"))
        keys = [os.path.basename(file[: -len(".nc")]) for file in files]
        return dict(zip(keys, files))

    def keys(self):
        return self._files().keys()

    def __getitem__(self, key):
        with self.fs.open(self._files()[key]) as f:
            return xr.open_dataset(f).load()

    def create_dir(self):
        self.fs.makedirs(self.path, exist_ok=True)


def _process_item(key, path, mapper):
    outputpath = os.path.join(path, f"{key}.nc")
    logger.info(f"saving {key} to {outputpath}")
    mapper[key].to_netcdf(outputpath)


def mapper_to_local(mapper: GeoMapper, path: str, threads: int = 10) -> LocalMapper:
    """Save a mapper to a local directory with multiprocessing

    Args:
        mapper: the mapper to save locally
        path: a "local" path. Really any path fsspec uses works.
        threads: number of processes to use for downloading

    Returns:
        a mapper representing the local data
    """

    local_mapper = LocalMapper(path)
    local_mapper.create_dir()

    with Pool(threads) as pool:
        pool.map(partial(_process_item, path=path, mapper=mapper), mapper.keys())

    return local_mapper
