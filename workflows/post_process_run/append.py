#!/usr/bin/env python3
from datetime import datetime
from glob import glob
import os
import json
import logging
import re
import shutil
import tempfile

import cftime
import click
import fsspec
import numpy as np
import zarr

from post_process import authenticate, upload_dir

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

TIMESTAMP_FORMAT = "%Y%m%d.%H%M%S"
XARRAY_DIM_NAMES_ATTR = "_ARRAY_DIMENSIONS"


def _set_time_units_like(source_store: zarr.Group, target_store: zarr.Group):
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


def _update_zarray_shape(var_path, axis, increment):
    zarray_path = os.path.join(var_path, ".zarray")
    with open(zarray_path) as f:
        zarray = json.load(f)
    zarray["shape"][axis] += increment
    with open(zarray_path, "w") as f:
        json.dump(zarray, f)


def _assert_n_shift_valid(array: zarr.array, axis: str, n_shift: int):
    """Ensure chunk size evenly divides n_shift"""
    chunk_size = array.chunks[axis]
    if n_shift % chunk_size != 0:
        raise ValueError(
            f"Desired shift must be a multiple of chunk size along {axis}. Got shift "
            f"{n_shift} and chunk size {chunk_size} for {array}."
        )


def _assert_chunks_match(source_group: zarr.Group, target_group: zarr.Group, dim: str):
    """Ensure chunks for source and target groups are valid for appending.
    
    Specifically:
        1. all arrays in source_group have corresponding arrays in target_group.
        2. chunk size is same for each array in source and target group.
        3. dim length is a multiple of chunk size for target group.
        
    In addition, log a warning if dim length is not a multiple of chunk size for source
    group."""
    for key, source_array in source_group.items():
        if key not in target_group:
            raise KeyError(
                f"Cannot append {source_array} because there is no corresponding array "
                f"in {target_group}."
            )
        if dim in source_array.attrs[XARRAY_DIM_NAMES_ATTR]:
            axis = source_array.attrs[XARRAY_DIM_NAMES_ATTR].index(dim)
            target_array = target_group[key]
            _assert_array_chunks_match(source_array, target_array, axis)


def _assert_array_chunks_match(source: zarr.array, target: zarr.array, axis: int):
    source_chunk_size = source.chunks[axis]
    target_chunk_size = target.chunks[axis]
    if source_chunk_size != target_chunk_size:
        raise ValueError(
            "Must have equal chunk size for source and target zarr when appending. "
            f"Got source chunk size {source_chunk_size} and target chunk size "
            f"{target_chunk_size} along axis {axis}."
        )
    source_axis_length = source.shape[axis]
    target_axis_length = target.shape[axis]
    if target_axis_length % target_chunk_size != 0:
        raise ValueError(
            "Length of append dimension for target array must be a multiple of "
            f"chunk size. Got length {target_axis_length} and chunk size "
            f"{target_chunk_size} for axis {axis} of {target}."
        )
    if source_axis_length % source_chunk_size != 0:
        logger.warning(
            f"Length of append dimension for {source} source array is not a multiple "
            "of chunk size. Resulting zarr store cannot be further appended to."
        )


def _get_dim_size(group: zarr.Group, dim: str):
    """Get length of dim, assuming it is same for all arrays that contain it"""
    for array in group.values():
        if dim in array.attrs[XARRAY_DIM_NAMES_ATTR]:
            axis = array.attrs[XARRAY_DIM_NAMES_ATTR].index(dim)
            return array.shape[axis]


def _shift_store(group: zarr.Group, dim: str, n_shift: int):
    """Shift local zarr store which represents an xarray dataset by n_shift along dim
    
    Args:
        group: zarr Group for an xarray dataset backed by a DirectoryStore
        dim: name of dimension of xarray dataset along which to shift zarr
        n_shift: how far to shift. The chunk size along dim of every array in group
            must evenly divide n_shift.

    Note:
        The zarr store represented by group will no longer be valid after this
        function is called since its chunks will not be listed starting at 0. It is
        intended that the output of this function be copied into another zarr store as
        a method of appending.
    """
    for array in group.values():
        if dim in array.attrs[XARRAY_DIM_NAMES_ATTR]:
            axis = array.attrs[XARRAY_DIM_NAMES_ATTR].index(dim)
            _shift_array(array, axis, n_shift)


def _shift_array(array: zarr.Array, axis: int, n_shift: int):
    """Rename files within array backed by DirectoryStore: e.g. 0.0 -> 1.0 if n_shift
    equals chunks[axis] and axis=0"""
    _assert_n_shift_valid(array, axis, n_shift)
    array_path = os.path.join(array.store.dir_path(), array.path)
    _update_zarray_shape(array_path, axis, n_shift)
    generic_chunk_filename = ".".join("*" * array.ndim)
    chunk_paths = glob(os.path.join(array_path, generic_chunk_filename))
    # go in reverse order to not overwrite existing chunks
    for chunk_path in sorted(chunk_paths, reverse=True):
        _rename_chunk(chunk_path, axis, n_shift, array.chunks[axis])


def _rename_chunk(chunk_path: str, axis: int, n_shift: int, chunk_size: int):
    head, tail = os.path.split(chunk_path)
    chunk_indices = tail.split(".")
    chunk_indices[axis] = str(int(chunk_indices[axis]) + n_shift // chunk_size)
    new_chunk_path = os.path.join(head, ".".join(chunk_indices))
    os.rename(chunk_path, new_chunk_path)


def _get_initial_timestamp(rundir: str) -> str:
    with open(os.path.join(rundir, "time_stamp.out")) as f:
        lines = f.readlines()
    start_date = datetime(*[int(d) for d in re.findall(r"\d+", lines[0])])
    return start_date.strftime(TIMESTAMP_FORMAT)


def append_zarr_along_time(
    source_path: str, target_path: str, fs: fsspec.AbstractFileSystem, dim: str = "time"
):
    """Append local zarr store at source_path to zarr store at target_path along time.
    
    Args:
        source_path: Local path to zarr store that represents an xarray dataset.
        target_path: Local or remote url for zarr store to be appended to.
        fs: Filesystem for target_path.
        dim: (optional) name of time dimension. Defaults to "time".

    Raises:
        ValueError: If the chunk size in time does not evenly divide length of time
            dimension for zarr stores at source_path.

    Warning:
        The zarr store as source_path will be modified in place.
    """

    consolidate = False
    if fs.exists(target_path):
        consolidate = True
        source_store = zarr.open(source_path, mode="r+")
        target_store = zarr.open_consolidated(fsspec.get_mapper(target_path))
        _assert_chunks_match(source_store, target_store, dim)
        _set_time_units_like(source_store, target_store)
        _shift_store(source_store, dim, _get_dim_size(target_store, dim))
    elif fs.protocol == "file":
        os.makedirs(target_path)

    upload_dir(source_path, target_path)

    if consolidate:
        zarr.consolidate_metadata(fsspec.get_mapper(target_path))


@click.command()
@click.argument("rundir")
@click.argument("destination")
@click.option("--segment_label", help="Defaults to timestamp of start of segment.")
def append_segment(rundir: str, destination: str, segment_label: str):
    """Append local RUNDIR to possibly existing output at DESTINATION
    
    Zarr's will be appended to in place, while all other files will be saved to
    DESTINATION/artifacts/SEGMENT_LABEL.
    """
    logger.info(f"Appending {rundir} to {destination}")
    authenticate()

    if not segment_label:
        segment_label = _get_initial_timestamp(rundir)

    fs, _, _ = fsspec.get_fs_token_paths(destination)

    with tempfile.TemporaryDirectory() as d_in:
        tmp_rundir = shutil.copytree(rundir, os.path.join(d_in, "rundir"))
        files = os.listdir(tmp_rundir)
        artifacts_dir = os.path.join(tmp_rundir, "artifacts", segment_label)
        os.makedirs(artifacts_dir, exist_ok=True)

        for file_ in files:
            tmp_rundir_file = os.path.join(tmp_rundir, file_)
            logger.info(f"Processing {tmp_rundir_file}")
            if file_.endswith(".zarr"):
                destination_file = os.path.join(destination, file_)
                logger.info(f"Appending {tmp_rundir_file} to {destination_file}")
                append_zarr_along_time(tmp_rundir_file, destination_file, fs)
                # remove temporary local copy so not uploaded twice
                shutil.rmtree(tmp_rundir_file)
            else:
                renamed_file = os.path.join(artifacts_dir, file_)
                os.rename(tmp_rundir_file, renamed_file)

        logger.info(f"Uploading non-zarr files from {tmp_rundir} to {destination}")
        upload_dir(tmp_rundir, destination)


if __name__ == "__main__":
    append_segment()
