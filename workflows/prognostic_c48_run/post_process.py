#!/usr/bin/env python
import os
import re
import shutil
from typing import Sequence, Iterable, Union
import xarray as xr
import tempfile
import subprocess
import logging
import click
from toolz import groupby
from itertools import chain

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

CHUNKS_2D = {"time": 96}
CHUNKS_3D = {"time": 8}

CHUNKS = {
    "diags.zarr": CHUNKS_2D,
    "atmos_dt_atmos.zarr": CHUNKS_2D,
    "sfc_dt_atmos.zarr": CHUNKS_2D,
    "atmos_8xdaily.zarr": CHUNKS_3D,
}


def upload_dir(d, dest):
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", d, dest])


def download_directory(dir_, dest):
    os.makedirs(dest, exist_ok=True)
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", dir_, dest])


def rechunk(ds, chunks):
    true_chunks = {}
    true_chunks.update(chunks)

    for dim in ds.dims:
        if dim not in chunks:
            true_chunks[dim] = len(ds[dim])
    return ds.chunk(true_chunks)


def authenticate():
    try:
        credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError:
        pass
    else:
        subprocess.check_call(
            ["gcloud", "auth", "activate-service-account", "--key-file", credentials]
        )


def clear_encoding(ds):
    ds.encoding = {}
    for variable in ds:
        ds[variable].encoding = {}


def parse_rundir(walker):
    """
    Args:
        walker: output of os.walk
    Returns:
        tiles, zarrs, other
    """
    tiles = []
    zarrs = []
    other = []
    for root, dirs, files in walker:
        for file_ in files:
            full_name = os.path.join(root, file_)
            if re.search(r"tile\d\.nc", file_):
                tiles.append(full_name)
            elif ".zarr" in root:
                pass
            else:
                other.append(full_name)

        search_path = []
        for dir_ in dirs:
            if dir_.endswith(".zarr"):
                zarrs.append(os.path.join(root, dir_))
            else:
                search_path.append(dir_)
        # only recurse into non-zarrs
        dirs[:] = search_path

    return tiles, zarrs, other


def open_tiles(tiles: Sequence[str]) -> Iterable[Union[str, xr.Dataset]]:
    grouped_tiles = groupby(lambda x: x[: -len(".tile1.nc")], tiles)
    for key, files in grouped_tiles.items():
        if key in CHUNKS:
            yield xr.open_mfdataset(
                sorted(files), concat_dim="tile", combine="nested"
            ).assign_attrs(path=key + ".zarr")
        else:
            for file in files:
                yield file


def open_zarrs(zarrs: Sequence[str]) -> Iterable[xr.Dataset]:
    for zarr in zarrs:
        yield xr.open_zarr(zarr).assign_attrs(path=zarr)


def process_item(item: Union[xr.Dataset, str], d_in: str, d_out: str):
    logger.info("Processing {item}")
    try:
        dest = os.path.join(d_out, os.path.relpath(item, d_in))  # type: ignore
    except TypeError:
        # is an xarray
        relpath = os.path.relpath(item.path, d_in)  # type: ignore
        chunks = CHUNKS.get(relpath, CHUNKS_2D)
        clear_encoding(item)
        chunked = rechunk(item, chunks)
        dest = os.path.join(d_out, relpath)
        chunked.to_zarr(dest, mode="w")
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(item, dest)  # type: ignore


@click.command()
@click.argument("rundir")
@click.argument("destination")
def post_process(rundir: str, destination: str):
    """Post-process the fv3gfs output located RUNDIR and save to DESTINATION

    Both RUNDIR and DESTINATION are URLs in GCS.

    This script rechunks the python zarr output and converts the netCDF
    outputs to zarr.
    """
    logger.info("Post-processing the run")
    authenticate()

    with tempfile.TemporaryDirectory() as d_in, tempfile.TemporaryDirectory() as d_out:

        if rundir.startswith("gs://"):
            download_directory(rundir, d_in)
        else:
            d_in = rundir

        tiles, zarrs, other = parse_rundir(os.walk(d_in))

        for item in chain(open_tiles(tiles), open_zarrs(zarrs), other):
            process_item(item, d_in, d_out)

        upload_dir(d_out, destination)


if __name__ == "__main__":
    post_process()
