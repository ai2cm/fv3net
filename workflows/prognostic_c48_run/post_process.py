import os
from dataclasses import dataclass
import argparse
import xarray as xr
import tempfile
import subprocess
import logging

logger = logging.getLogger(__file__)

CHUNKS_2D = {"time": 100}
CHUNKS_ATMOS = {"time": 100}
CHUNKS_SFC = {"time": 100}


def open_tiles(pattern):
    return xr.open_mfdataset(f"{pattern}.tile?.nc", concat_dim="tile", combine="nested")


paths_openers = [
    ("diags.zarr", "diags.zarr", xr.open_zarr, CHUNKS_2D),
    ("atmos_dt_atmos.zarr", "atmos_dt_atmos", open_tiles, CHUNKS_ATMOS),
    ("sfc_dt_atmos.zarr", "sfc_dt_atmos", open_tiles, CHUNKS_SFC),
    ("atmos_8xdaily.zarr", "atmos_8xdaily", open_tiles, CHUNKS_ATMOS),
    ("atmos_static.zarr", "atmos_static", open_tiles, {}),
]

files_to_copy = [
    "stderr.log",
    "stdout.log",
    "input.nml",
    "atmos_8xdaily.tile1.nc",
    "atmos_8xdaily.tile2.nc",
    "atmos_8xdaily.tile3.nc",
    "atmos_8xdaily.tile4.nc",
    "atmos_8xdaily.tile5.nc",
    "atmos_8xdaily.tile6.nc",
    "atmos_dt_atmos.tile1.nc",
    "atmos_dt_atmos.tile2.nc",
    "atmos_dt_atmos.tile3.nc",
    "atmos_dt_atmos.tile4.nc",
    "atmos_dt_atmos.tile5.nc",
    "atmos_dt_atmos.tile6.nc",
    "atmos_static.tile1.nc",
    "atmos_static.tile2.nc",
    "atmos_static.tile3.nc",
    "atmos_static.tile4.nc",
    "atmos_static.tile5.nc",
    "atmos_static.tile6.nc",
    "sfc_dt_atmos.tile1.nc",
    "sfc_dt_atmos.tile2.nc",
    "sfc_dt_atmos.tile3.nc",
    "sfc_dt_atmos.tile4.nc",
    "sfc_dt_atmos.tile5.nc",
    "sfc_dt_atmos.tile6.nc",
]

directories_to_copy = ["job_config/", "INPUT/", "RESTART/"]


def rechunk(ds, chunks):
    true_chunks = {}
    true_chunks.update(chunks)

    for dim in ds.dims:
        if dim not in chunks:
            true_chunks[dim] = len(ds[dim])
    return ds.chunk(true_chunks)


def upload_dir(d, dest):
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", d, dest])


def copy_files(files, dest):
    subprocess.check_call(["gsutil", "-m", "cp"] + files + [dest])


def clear_encoding(ds):
    ds.encoding = {}
    for variable in ds:
        ds[variable].encoding = {}


def post_process(rundir, dest):
    logger.info("Post-processing the run")

    paths = [os.path.join(rundir, file) for file in files_to_copy]
    copy_files(paths, dest)

    for dir_ in directories_to_copy:
        path = os.path.join(rundir, dir_)
        upload_dir(path, os.path.join(dest, dir_))

    with tempfile.TemporaryDirectory() as d:
        for output, in_, opener, chunks in paths_openers:
            logger.info(f"Processing {in_}")
            path = os.path.join(rundir, in_)
            ds = opener(path)
            clear_encoding(ds)
            chunked = rechunk(ds, chunks)
            path_out = os.path.join(d, output)
            chunked.to_zarr(path_out, consolidated=True)

        upload_dir(d, dest)
