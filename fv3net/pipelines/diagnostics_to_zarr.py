import logging
import os
import argparse
import tempfile

import apache_beam as beam
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
import fsspec

logger = logging.getLogger(__name__)

INITIAL_CHUNKS = {"time": 12}
TILES = range(1, 7)
COMMON_SUFFIX = ".tile1.nc"


def run(args, pipeline_args):
    rundir = args.rundir
    if args.diagnostic_dir is None:
        diagnostic_dir = os.path.join(_get_parent_dir(rundir), "diagnostic_zarr")
    else:
        diagnostic_dir = args.diagnostic_dir
    if args.diagnostic_categories is None:
        diagnostic_categories = _get_all_diagnostic_categories(rundir, _get_fs(rundir))
    else:
        diagnostic_categories = args.diagnostic_categories

    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | beam.Create(diagnostic_categories)
            | "OpenConvertSave"
            >> beam.ParDo(
                open_convert_save, rundir=rundir, diagnostic_dir=diagnostic_dir
            )
        )


def _get_fs(path):
    """Return the fsspec filesystem required to handle a given path."""
    if path.startswith("gs://"):
        return fsspec.filesystem("gs")
    else:
        return fsspec.filesystem("file")


def _get_all_diagnostic_categories(rundir, fs):
    """ get full paths for all files in rundir that end in COMMON_SUFFIX """
    full_paths = fs.glob(os.path.join(rundir, f"*{COMMON_SUFFIX}"))
    return [_get_category_from_path(path) for path in full_paths]


def _get_category_from_path(path):
    """ get part of filename before COMMON_SUFFIX """
    basename = os.path.basename(path)
    return basename[:-len(COMMON_SUFFIX)]


def _get_parent_dir(path):
    if path[-1] == "/":
        path = path[:-1]
    return os.path.split(path)[0]


def open_convert_save(diagnostic_category, rundir, diagnostic_dir):
    logger.info(f"Converting {diagnostic_category} to zarr")
    remote_zarr = os.path.join(diagnostic_dir, diagnostic_category)
    fs = _get_fs(remote_zarr)
    # cannot read and write at same time with fsspec, so must save
    # zarr locally before uploading to GCS
    with tempfile.TemporaryDirectory() as local_zarr:
        for tile in TILES:
            logger.info(f"tile {tile}")
            prefix = os.path.join(rundir, diagnostic_category)
            remote_nc = f"{prefix}.tile{tile}.nc"
            with fsspec.open(remote_nc) as nc:
                xr.open_dataset(nc, chunks=INITIAL_CHUNKS).assign_coords(
                    {"tile": tile - 1}
                ).expand_dims("tile").to_zarr(local_zarr, append_dim="tile")
        fs.put(local_zarr, remote_zarr, recursive=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rundir",
        type=str,
        required=True,
        help="Location of run directory. May be local or remote path.",
    )
    parser.add_argument(
        "--diagnostic-dir",
        type=str,
        default=None,
        help="Location to save zarr stores. Defaults to the parent of rundir.",
    )
    parser.add_argument(
        "--diagnostic-categories",
        type=str,
        default=None,
        nargs="+",
        help="Optionally specify one or more categories of diagnostic files. Provide "
        "part of filename before .tile*.nc. Defaults to all diagnostic categories in "
        "rundir.",
    )
    args, pipeline_args = parser.parse_known_args()

    run(args, pipeline_args)
