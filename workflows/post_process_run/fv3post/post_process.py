#!/usr/bin/env python3
import os
import re
import yaml
import shutil
from typing import Sequence, Iterable, Union, Mapping
import numpy as np
import xarray as xr
import tempfile
import logging
import click
from toolz import groupby
from itertools import chain
from .gsutil import authenticate, upload_dir, download_directory

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

ChunkSpec = Mapping[str, Mapping[str, int]]
CHUNKS_DEFAULT = {"time": 96}


def _get_true_chunks(ds, chunks):
    true_chunks = {}
    true_chunks.update(chunks)

    for dim in ds.dims:
        if dim not in chunks:
            true_chunks[dim] = len(ds[dim])
    return true_chunks


def rechunk(ds, chunks):
    true_chunks = _get_true_chunks(ds, chunks)
    return ds.chunk(true_chunks)


def encode_chunks(ds, chunks):
    """Ensure zarr-stores are saved to disk with desired chunk sizes"""
    true_chunks = _get_true_chunks(ds, chunks)
    for variable in set(ds.data_vars) | set(ds.coords):
        variable_chunks = [true_chunks[dim] for dim in ds[variable].dims]
        ds[variable].encoding["chunks"] = variable_chunks
    return ds


def clear_encoding(ds):
    ds.encoding = {}
    for variable in ds:
        ds[variable].encoding = {}


def cast_time(ds):
    if "time" in ds.coords:
        try:
            # explicitly set time dtype to avoid invalid type promotion error
            ds = ds.assign_coords(time=ds.time.astype(np.datetime64))
        except TypeError:
            pass  # if cannot cast to np.datetime64, leave time axis alone
    return ds


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


def open_tiles(
    tiles: Sequence[str], base: str, chunks: ChunkSpec
) -> Iterable[Union[str, xr.Dataset]]:
    grouped_tiles = groupby(lambda x: x[: -len(".tile1.nc")], tiles)
    for key, files in grouped_tiles.items():
        path = key + ".zarr"
        relpath = os.path.relpath(path, base)
        if relpath in chunks:
            yield xr.open_mfdataset(
                sorted(files), concat_dim="tile", combine="nested"
            ).assign_attrs(path=path)
        else:
            for file in files:
                yield file


def open_zarrs(zarrs: Sequence[str]) -> Iterable[xr.Dataset]:
    for zarr in zarrs:
        yield xr.open_zarr(zarr).assign_attrs(path=zarr)


def process_item(
    item: Union[xr.Dataset, str], d_in: str, d_out: str, chunks: ChunkSpec,
):
    logger.info(f"Processing {item}")
    try:
        dest = os.path.join(d_out, os.path.relpath(item, d_in))  # type: ignore
    except TypeError:
        # is an xarray
        relpath = os.path.relpath(item.path, d_in)  # type: ignore
        chunks = chunks.get(relpath, CHUNKS_DEFAULT)
        clear_encoding(item)
        chunked = rechunk(item, chunks)
        chunked = encode_chunks(chunked, chunks)
        dest = os.path.join(d_out, relpath)
        chunked = cast_time(chunked)
        chunked.to_zarr(dest, mode="w", consolidated=True)
    except ValueError:
        # is an empty xarray, do nothing
        logger.warning(f"Skipping {item} since it is an empty dataset.")
        pass
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            shutil.copy(item, dest)  # type: ignore
        except FileNotFoundError:
            logger.warning(f"{item} not found. Possibly a broken symlink.")


@click.command()
@click.argument("rundir")
@click.argument("destination")
@click.option(
    "--chunks", type=click.Path(), help="path to yaml file containing chunk information"
)
def post_process(rundir: str, destination: str, chunks: str):
    """Post-process the fv3gfs output located RUNDIR and save to DESTINATION

    RUNDIR and DESTINATION may be local or GCS paths.

    This script rechunks the python zarr output and converts the netCDF
    outputs to zarr.
    """
    logger.info("Post-processing the run")
    authenticate()

    if chunks:
        with open(chunks) as f:
            chunks = yaml.safe_load(f)
    else:
        chunks = {}

    with tempfile.TemporaryDirectory() as d_in, tempfile.TemporaryDirectory() as d_out:

        if rundir.startswith("gs://"):
            download_directory(rundir, d_in)
        else:
            d_in = rundir

        if not destination.startswith("gs://"):
            d_out = destination

        tiles, zarrs, other = parse_rundir(os.walk(d_in, topdown=True))

        for item in chain(open_tiles(tiles, d_in, chunks), open_zarrs(zarrs), other):
            process_item(item, d_in, d_out, chunks)

        if destination.startswith("gs://"):
            upload_dir(d_out, destination)


if __name__ == "__main__":
    post_process()
