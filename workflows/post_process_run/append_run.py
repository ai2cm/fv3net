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


def set_time_units_like(source_store: zarr.Group, target_store: zarr.Group):
    """Modify all time-like variables in source_store to use same units as
    corresponding variable in target_store. The provided source_store must be
    opened in a mode such that it can be modified (e.g. mode='r+')"""
    for variable, source_array in source_store.items():
        target_array = target_store[variable]
        if "units" in source_array.attrs and "since" in source_array.attrs["units"]:
            _set_array_time_units_like(source_array, target_array)


def _set_array_time_units_like(source_array: zarr.Array, target_array: zarr.Array):
    _assert_calendars_same(source_array, target_array)
    source_array[:] = _rebase_times(
        source_array[:],
        source_array.attrs["units"],
        source_array.attrs["calendar"],
        target_array.attrs["units"],
    )
    source_array.attrs["units"] = target_array.attrs["units"]


def _assert_calendars_same(source_array: zarr.Array, target_array: zarr.Array):
    if "calendar" not in source_array.attrs:
        raise AttributeError(
            f"Source array {source_array} missing calendar. Cannot rebase times."
        )
    if "calendar" not in target_array.attrs:
        raise AttributeError(
            f"Target array {target_array} missing calendar. Cannot rebase times."
        )
    if source_array.attrs["calendar"] != target_array.attrs["calendar"]:
        raise ValueError(
            "Calendars must be the same between source and target arrays to set "
            "time units to be the same."
        )


def _rebase_times(
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
            dest_item = os.path.join(destination, item)
            if fs.exists(dest_item):
                source_store = zarr.open(rundir_item, mode="r+")
                target_store = zarr.open_consolidated(fsspec.get_mapper(dest_item))
                set_time_units_like(source_store, target_store)
                shift_store(rundir_item, "time", target_store["time"].size)
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
