import logging
import os
import argparse
import tempfile

import apache_beam as beam
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
import fsspec

from vcm.cloud import gsutil

logger = logging.getLogger(__name__)

INITIAL_CHUNKS = {"time": 12}
TILES = range(1, 7)
PARENT_BUCKET = "gs://vcm-ml-data/2019-12-12-baseline-FV3GFS-runs/"
DIAGNOSTIC_CATEGORIES = [
    # "atmos_8xdaily",
    # "tracers_8xdaily",
    # "nudge_dt_8xdaily",
    # "atmos_hourly",
    "sfc_hourly"
]


def run(args, pipeline_args):
    rundir = args.rundir
    if args.diagnostic_dir is None:
        diagnostic_dir = os.path.join(_get_parent_dir(rundir), "diagnostic_zarr")
    else:
        diagnostic_dir = args.diagnostic_dir

    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | beam.Create(args.diagnostic_categories)
            | "OpenConvertSave"
            >> beam.ParDo(
                open_convert_save, rundir=rundir, diagnostic_dir=diagnostic_dir
            )
        )


def _get_parent_dir(path):
    if path[-1] == "/":
        path = path[:-1]
    return os.path.split(path)[0]


def open_convert_save(diagnostic_category, rundir, diagnostic_dir):
    logger.info(f"Converting {diagnostic_category} to zarr")
    remote_zarr = os.path.join(diagnostic_dir, diagnostic_category)
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
        gsutil.copy(local_zarr, remote_zarr)


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
        help="Location to save zarr store. Defaults to the parent of rundir.",
    )
    parser.add_argument(
        "--diagnostic-categories",
        type=str,
        default=DIAGNOSTIC_CATEGORIES,
        nargs="+",
        help="One or more categories of diagnostic files. Provide everything before "
        ".tile*.nc. Defaults to atmos_8xdaily, tracers_8xdaily, nudge_dt_8xdaily, "
        "atmos_hourly, sfc_hourly.",
    )
    args, pipeline_args = parser.parse_known_args()

    run(args, pipeline_args)
