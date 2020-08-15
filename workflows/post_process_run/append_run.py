import glob
import os
import json
import re
import shutil
from typing import Union

import click
import fsspec
import xarray as xr

from post_process import (
    authenticate,
    parse_rundir,
    post_process,
    ChunkSpec,
    get_chunks,
    upload_dir,
)


def increment_zarray_item(var_path, ax, increment, item):
    zarray_path = os.path.join(var_path, ".zarray")
    with open(zarray_path) as f:
        zarray = json.load(f)
    zarray[item][ax] += increment
    with open(zarray_path, "w") as f:
        json.dump(zarray, f)


def _shift_chunks(path: str, dim: str, n: int, consolidated=True):
    """Shift local zarr store at path by n chunks along dim"""
    ds = xr.open_zarr(path, consolidated=consolidated)

    if "dim" in ds.coords:
        increment_zarray_item(os.path.join(path, dim), 0, 4, "shape")
        increment_zarray_item(os.path.join(path, dim), 0, 4, "chunks")

    for var in ds.data_vars:
        da = ds[var]
        var_path = os.path.join(path, var)
        ax = da.get_axis_num(dim)
        increment_zarray_item(var_path, ax, n * da.chunks[ax][0], "shape")
        chunk_glob_str = ".".join("*" * len(da.sizes))
        for chunk_path in glob.glob(os.path.join(var_path, chunk_glob_str)):
            head, tail = os.path.split(chunk_path)
            chunk_positions = re.findall(r"[0-9]+", tail)
            chunk_positions[ax] = str(int(chunk_positions[ax]) + n)
            new_chunk_path = os.path.join(head, ".".join(chunk_positions))
            shutil.move(chunk_path, new_chunk_path)


def append_item(
    item: Union[xr.Dataset, str], d_in: str, d_out: str, chunks: ChunkSpec, step: str
):
    logger.info(f"Processing {item}")
    try:
        dest = os.path.join(d_out, os.path.relpath(item, d_in), step)  # type: ignore
    except TypeError:
        # is an xarray
        relpath = os.path.relpath(item.path, d_in)  # type: ignore
        chunks = chunks.get(relpath, CHUNKS_2D)
        if "time" in chunks:
            if item.sizes["time"] % chunks["time"] != 0:
                raise ValueError("Time chunk size must evenly divide total length")
        clear_encoding(item)
        chunked = rechunk(item, chunks)
        dest = os.path.join(d_out, relpath)
        chunked = cast_time(chunked)
        chunked.to_zarr(dest, mode="w", consolidated=True)
        num_time_chunks = int(chunked.sizes["time"] / chunked.chunks["time"])
        _shift_chunks(dest, "time", num_time_chunks * step)
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            shutil.copy(item, dest)  # type: ignore
        except FileNotFoundError:
            logger.warning(f"{item} not found. Possibly a broken symlink.")


@click.command()
@click.argument("rundir")
@click.argument("destination")
@click.argument("step", type=int)
@click.option(
    "--chunks", type=click.Path(), help="path to yaml file containing chunk information"
)
def append_run(rundir: str, destination: str, chunks: str):
    """Post-process the fv3gfs output located at RUNDIR and append to possibly
    existing DESTINATION"""
    logger.info("Post-processing and appending the run")
    authenticate()

    if chunks:
        with open(chunks) as f:
            user_chunks = yaml.safe_load(f)
    else:
        user_chunks = {}
    chunks = get_chunks(user_chunks)

    with tempfile.TemporaryDirectory() as d_in, tempfile.TemporaryDirectory() as d_out:

        if rundir.startswith("gs://"):
            download_directory(rundir, d_in)
        else:
            d_in = rundir

        tiles, zarrs, other = parse_rundir(os.walk(d_in, topdown=True))

        for item in chain(open_tiles(tiles, d_in, chunks), open_zarrs(zarrs), other):
            append_item(item, d_in, d_out, chunks)

        upload_dir(d_out, destination)


if __name__ == "__main__":
    append_run()
