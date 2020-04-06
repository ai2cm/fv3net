import logging
import os
import argparse
import tempfile

import apache_beam as beam
import xarray as xr
import zarr
from apache_beam.options.pipeline_options import PipelineOptions
import fsspec

from vcm.cloud import gsutil

logger = logging.getLogger(__name__)

INITIAL_CHUNKS = {"time": 192}
TILES = range(1, 7)
COMMON_SUFFIX = ".tile1.nc"


def run(args, pipeline_args):
    rundir = args.rundir
    diagnostic_dir = _parse_diagnostic_dir(args.diagnostic_dir, rundir)
    diagnostic_categories = _parse_categories(args.diagnostic_categories, rundir)
    logger.info(f"Diagnostic zarrs being written to {diagnostic_dir}")
    logger.info(f"Diagnostic categories to convert are {diagnostic_categories}")
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


def open_convert_save(diagnostic_category, rundir, diagnostic_dir):
    remote_zarr = os.path.join(diagnostic_dir, f"{diagnostic_category}.zarr")
    with tempfile.TemporaryDirectory() as local_zarr:
        for tile in TILES:
            logger.info(f"Converting category {diagnostic_category} tile {tile}")
            remote_nc = os.path.join(rundir, f"{diagnostic_category}.tile{tile}.nc")
            with fsspec.open(remote_nc) as nc:
                xr.open_dataset(nc, chunks=INITIAL_CHUNKS).assign_coords(
                    {"tile": tile - 1}
                ).expand_dims("tile").to_zarr(local_zarr, append_dim="tile")
        zarr.convenience.consolidate_metadata(local_zarr)
        logger.info(f"Starting upload of complete zarr for {diagnostic_category}")
        # fsspec is slow at copying many files, so use gsutil to copy zarr store
        if not remote_zarr.startswith("gs://"):
            os.makedirs(remote_zarr, exists_ok=True)
        gsutil.copy(local_zarr, remote_zarr)
        logger.info(f"Finished upload of complete zarr for {diagnostic_category}")


def _parse_categories(diagnostic_categories, rundir):
    if diagnostic_categories is None:
        return _get_all_diagnostic_categories(rundir, _get_fs(rundir))
    else:
        return diagnostic_categories


def _parse_diagnostic_dir(diagnostic_dir, rundir):
    if diagnostic_dir is None:
        return rundir
    else:
        return diagnostic_dir


def _get_all_diagnostic_categories(rundir, fs):
    """ get full paths for all files in rundir that end in COMMON_SUFFIX """
    full_paths = fs.glob(os.path.join(rundir, f"*{COMMON_SUFFIX}"))
    return [_get_category_from_path(path) for path in full_paths]


def _get_category_from_path(path):
    """ get part of filename before COMMON_SUFFIX """
    basename = os.path.basename(path)
    return basename[: -len(COMMON_SUFFIX)]


def _get_fs(path):
    """Return the fsspec filesystem required to handle a given path."""
    if path.startswith("gs://"):
        return fsspec.filesystem("gs")
    else:
        return fsspec.filesystem("file")


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
        help="Location to save zarr stores. Defaults to rundir.",
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
