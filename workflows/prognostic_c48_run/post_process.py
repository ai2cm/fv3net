#!/usr/bin/env python
import os
import xarray as xr
import tempfile
import subprocess
import logging
import click

logger = logging.getLogger(__file__)

CHUNKS_2D = {"time": 96}
CHUNKS_3D = {"time": 8}


def open_tiles(pattern):
    return xr.open_mfdataset(f"{pattern}.tile?.nc", concat_dim="tile", combine="nested")


paths_openers = [
    ("diags.zarr", "diags.zarr", xr.open_zarr, CHUNKS_2D),
    ("atmos_dt_atmos.zarr", "atmos_dt_atmos", open_tiles, CHUNKS_2D),
    ("sfc_dt_atmos.zarr", "sfc_dt_atmos", open_tiles, CHUNKS_2D),
    ("atmos_8xdaily.zarr", "atmos_8xdaily", open_tiles, CHUNKS_3D),
]

files_to_copy = [
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
    "sfc_dt_atmos.tile1.nc",
    "sfc_dt_atmos.tile2.nc",
    "sfc_dt_atmos.tile3.nc",
    "sfc_dt_atmos.tile4.nc",
    "sfc_dt_atmos.tile5.nc",
    "sfc_dt_atmos.tile6.nc",
]

directories_to_copy = ["job_config/", "INPUT/", "RESTART/"]

directories_to_download = ["diags.zarr"]


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
    subprocess.check_call(["gsutil", "-m", "cp", "-r"] + files + [dest])


def download_directory(dir_, dest):
    os.makedirs(dest, exist_ok=True)
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", dir_, dest])


def clear_encoding(ds):
    ds.encoding = {}
    for variable in ds:
        ds[variable].encoding = {}


def download_rundir(rundir: str, output: str):

    os.makedirs(output, exist_ok=True)

    # download directories
    for dir_ in directories_to_download:
        path_in = os.path.join(rundir, dir_)
        path_out = os.path.join(output, dir_)
        download_directory(path_in, path_out)
        assert os.path.isdir(path_out), path_out

    # download files
    paths = [os.path.join(rundir, file) for file in files_to_copy]
    copy_files(paths, output)


def convert_data(input_dir: str, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)

    # process zarrs
    for output, in_, opener, chunks in paths_openers:
        logger.info(f"Processing {in_}")
        path = os.path.join(input_dir, in_)
        ds = opener(path)
        clear_encoding(ds)
        chunked = rechunk(ds, chunks)
        path_out = os.path.join(output_dir, output)
        chunked.to_zarr(path_out, consolidated=True, mode="w")


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

    for dir_ in directories_to_copy:
        path = os.path.join(rundir, dir_)
        upload_dir(path, os.path.join(destination, dir_))

    with tempfile.TemporaryDirectory() as d_in, tempfile.TemporaryDirectory() as d_out:
        download_rundir(rundir, d_in)
        convert_data(d_in, d_out)
        upload_dir(d_out, destination)


if __name__ == "__main__":
    post_process()
