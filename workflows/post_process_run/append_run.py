from glob import glob
import os
import json
import re
import shutil
from typing import Union, Sequence

import click
import fsspec
import xarray as xr
import zarr

from post_process import (
    authenticate,
    parse_rundir,
    post_process,
    ChunkSpec,
    get_chunks,
    upload_dir,
)


def _update_zarray_shape(var_path, ax, increment):
    zarray_path = os.path.join(var_path, ".zarray")
    with open(zarray_path) as f:
        zarray = json.load(f)
    zarray["shape"][ax] += increment
    with open(zarray_path, "w") as f:
        json.dump(zarray, f)


def _assert_chunks_valid(array: zarr.array, ax: str, n_shift: int):
    """Ensure chunk size evenly divides array length along ax and evenly divides
    n_shift"""
    chunk_size = array.chunks[ax]
    if array.shape[ax] % chunk_size != 0:
        raise ValueError(f"Chunks are not uniform along axis {ax} for {array}")
    if n_shift % chunk_size != 0:
        raise ValueError(
            f"Desired shift must be a multiple of chunk size along {ax}. Got shift "
            f"{n_shift} and chunk size {chunk_size} for {array}."
        )


def _shift_store(path: str, dim: str, n_shift: int):
    """Shift consolidated local zarr store by n_shift along dim. Chunk size must be
    uniform for each variable and it must evenly divide n_shift."""
    store = zarr.open_consolidated(path)
    for variable, array in store.items():
        if dim in array.attrs["_ARRAY_DIMENSIONS"]:
            ax = array.attrs["_ARRAY_DIMENSIONS"].index(dim)
            _shift_array(path, array, ax, n_shift)


def _shift_array(store_path: str, array: zarr.array, ax: int, n_shift: int):
    _assert_chunks_valid(array, ax, n_shift)
    array_path = os.path.join(store_path, array.path)
    _update_zarray_shape(array_path, ax, n_shift)
    generic_chunk_filename = _chunk_filename("*" * array.ndim)
    for chunk_path in glob(os.path.join(array_path, generic_chunk_filename)):
        _move_chunk(chunk_path, ax, n_shift, array.chunks[ax])


def _move_chunk(chunk_path, ax, n_shift, chunk_size):
    head, tail = os.path.split(chunk_path)
    chunk_indices = re.findall(r"[0-9]+", tail)
    chunk_indices[ax] = str(int(chunk_indices[ax]) + n_shift // chunk_size)
    new_chunk_path = os.path.join(head, _chunk_filename(chunk_indices))
    shutil.move(chunk_path, new_chunk_path)


def _chunk_filename(chunk_indices: Union[str, Sequence[str]]):
    return ".".join(chunk_indices)


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
                raise ValueError(
                    "Desired time chunk size must evenly divide total length"
                )
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
