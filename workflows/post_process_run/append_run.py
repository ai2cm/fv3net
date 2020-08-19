#!/usr/bin/env python3
from datetime import datetime
from glob import glob
import os
import json
import logging
import re
from typing import Union, Sequence

import cftime
import click
import fsspec
import numpy as np
import xarray as xr
import zarr

from post_process import authenticate, upload_dir

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

TIMESTAMP_FORMAT = "%Y%m%d.%H%M%S"


def encode_time_units_like(source_path: str, target_path: str):
    source_store = zarr.open(source_path, mode="r+")
    target_store = zarr.open_consolidated(fsspec.get_mapper(target_path))
    if source_store["time"].attrs["calendar"] != target_store["time"].attrs["calendar"]:
        raise ValueError("Calendars must be the same to encode same time units.")
    source_store["time"][:] = rebase_times(
        source_store["time"][:],
        source_store["time"].attrs["units"],
        target_store["time"].attrs["calendar"],
        target_store["time"].attrs["units"],
    )
    source_store["time"].attrs["units"] = target_store["time"].attrs["units"]


def rebase_times(
    values: np.ndarray, input_units: str, calendar: str, output_units: str
) -> np.ndarray:
    dates = cftime.num2date(values, input_units, calendar)
    return cftime.date2num(dates, output_units, calendar)


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


def shift_store(path: str, dim: str, n_shift: int):
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
        _rename_chunk(chunk_path, ax, n_shift, array.chunks[ax])


def _rename_chunk(chunk_path, ax, n_shift, chunk_size):
    head, tail = os.path.split(chunk_path)
    chunk_indices = re.findall(r"[0-9]+", tail)
    chunk_indices[ax] = str(int(chunk_indices[ax]) + n_shift // chunk_size)
    new_chunk_path = os.path.join(head, _chunk_filename(chunk_indices))
    os.rename(chunk_path, new_chunk_path)


def _chunk_filename(chunk_indices: Union[str, Sequence[str]]):
    return ".".join(chunk_indices)


def _get_initial_timestamp(rundir: str) -> str:
    with open(os.path.join(rundir, "time_stamp.out")) as f:
        lines = f.readlines()
    start_date = datetime(*[int(d) for d in re.findall(r"\d+", lines[0])])
    return start_date.strftime(TIMESTAMP_FORMAT)


@click.command()
@click.argument("rundir")
@click.argument("destination")
@click.option("--segment_label", help="Defaults to timestamp of start of segment.")
def append_run(rundir: str, destination: str, segment_label: str):
    """Given post-processed fv3gfs output located at local path RUNDIR, append to
    possibly existing GCS url DESTINATION. Zarr's will be appended to in place, while
    all other files will be saved to DESTINATION/artifacts/SEGMENT_LABEL."""
    logger.info(f"Appending {rundir} to {destination}")
    authenticate()

    if not segment_label:
        segment_label = _get_initial_timestamp(rundir)

    fs, _, _ = fsspec.get_fs_token_paths(destination)

    rundir = os.path.abspath(rundir)
    items = os.listdir(rundir)
    artifacts_dir = os.path.join(rundir, "artifacts", segment_label)
    os.makedirs(artifacts_dir, exist_ok=True)

    zarrs_to_consolidate = []
    for item in items:
        rundir_item = os.path.join(rundir, item)
        if item.endswith(".zarr"):
            destination_item = os.path.join(destination, item)
            if fs.exists(destination_item):
                encode_time_units_like(rundir_item, destination_item)
                mapper = fsspec.get_mapper(destination_item)
                shift = xr.open_zarr(mapper, consolidated=True).sizes["time"]
                shift_store(rundir_item, "time", shift)
            zarrs_to_consolidate.append(item)
        else:
            renamed_item = os.path.join(artifacts_dir, item)
            os.rename(rundir_item, renamed_item)

    upload_dir(rundir, destination)

    for item in zarrs_to_consolidate:
        mapper = fsspec.get_mapper(os.path.join(destination, item))
        zarr.consolidate_metadata(mapper)


if __name__ == "__main__":
    append_run()
