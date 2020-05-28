import apache_beam as beam
import tempfile
import os
import logging
import gcsfs
from pathlib import Path
import dask
from typing import Mapping

import pandas as pd
import xarray as xr

from apache_beam.io.filesystems import FileSystems

from apache_beam.options.pipeline_options import PipelineOptions

from fv3net.pipelines.common import list_timesteps, FunctionSource
import vcm
from vcm.cloud import gcs
from vcm import parse_timestep_str_from_path

logger = logging.getLogger("CoarsenPipeline")
logger.setLevel(logging.DEBUG)

NUM_FILES_IN_COARSENED_DIR = 24


def check_coarsen_incomplete(gcs_url, output_prefix):

    timestep = parse_timestep_str_from_path(gcs_url)
    output_timestep_dir = os.path.join(output_prefix, timestep)

    fs = gcsfs.GCSFileSystem()
    timestep_exists = fs.exists(output_timestep_dir)

    if timestep_exists:
        timestep_files = fs.ls(output_timestep_dir)
        incorrect_num_files = len(timestep_files) != NUM_FILES_IN_COARSENED_DIR
        return incorrect_num_files
    else:
        return True


def _copy_gridspec(local_spec_dir, gridspec_gcs_path):

    os.makedirs(local_spec_dir, exist_ok=True)
    logger.debug(f"Copying gridspec to {local_spec_dir}")
    gcs.download_all_bucket_files(
        gridspec_gcs_path, local_spec_dir, include_parent_in_stem=False
    )


def _copy_restart(local_timestep_dir, timestep_gcs_path):

    os.makedirs(local_timestep_dir, exist_ok=True)
    logger.debug(f"Copying restart files to {local_timestep_dir}")
    gcs.download_all_bucket_files(timestep_gcs_path, local_timestep_dir)


def _open_grid_spec(grid_spec_prefix):
    tiles = pd.Index(range(6), name="tile")
    filename = grid_spec_prefix + ".tile*.nc"
    return xr.open_mfdataset(filename, concat_dim=[tiles], combine="nested")


def output_filename(directory: str, time: str, category: str, tile: int) -> str:
    assert tile in [1, 2, 3, 4, 5, 6]
    return os.path.join(directory, time, f"{time}.{category}.tile{tile}.nc")


def save_data(time: str, category: str, coarsen: xr.Dataset, directory: str = "."):
    for tile in range(6):
        data = coarsen.isel(tile=tile)
        filename = output_filename(directory, time, category, tile + 1)
        logger.info(f"Saving to {filename}")

        try:
            FileSystems.mkdirs(os.path.dirname(filename))
        except IOError:
            pass

        with FileSystems.create(filename) as f:
            vcm.dump_nc(data, f)


def _open_restart_categories(time: str, prefix: str) -> Mapping[str, xr.Dataset]:
    source = {}

    OUTPUT_CATEGORY_NAMES = {
        "fv_core.res": "fv_core_coarse.res",
        "fv_srf_wnd.res": "fv_srf_wnd_coarse.res",
        "fv_tracer.res": "fv_tracer_coarse.res",
        "sfc_data": "sfc_data_coarse",
    }

    for output_category in OUTPUT_CATEGORY_NAMES:
        input_category = OUTPUT_CATEGORY_NAMES[output_category]
        tile_prefix = os.path.join(prefix, time, f"{time}.{input_category}")
        source[output_category] = vcm.open_tiles(tile_prefix)
    return source


def coarsen_timestep(
    curr_timestep, coarsen_factor: int, grid_spec: xr.Dataset, prefix="."
):
    tmp_timestep_dir = os.path.join(prefix, "local_fine_dir")
    source = _open_restart_categories(curr_timestep, tmp_timestep_dir)

    # import computation
    for category, data in vcm.coarsen_restarts_on_pressure(
        coarsen_factor, grid_spec, source
    ).items():
        yield curr_timestep, category, data


def run(args, pipeline_args=None):
    logging.basicConfig(level=logging.DEBUG)

    gridspec_path = os.path.join(args.gcs_grid_spec_path, "grid_spec")
    source_resolution = args.source_resolution
    target_resolution = args.target_resolution

    output_dir_prefix = args.gcs_dst_dir
    if args.add_target_subdir:
        output_dir_prefix = os.path.join(output_dir_prefix, f"C{target_resolution}")

    coarsen_factor = source_resolution // target_resolution

    timesteps = ["20160801.001500"]

    prefix = "test-outputs"

    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        grid_spec = p | FunctionSource(
            lambda x: vcm.open_tiles(x).load(), gridspec_path
        )
        (
            p
            | "CreateTStepURLs" >> beam.Create(timesteps)
            # | "CheckCompleteTSteps"
            # >> beam.Filter(check_coarsen_incomplete, output_dir_prefix)
            | "CoarsenTStep"
            >> beam.ParDo(
                coarsen_timestep,
                coarsen_factor=coarsen_factor,
                grid_spec=beam.pvalue.AsSingleton(grid_spec),
                # TODO remove this argument
                prefix=prefix,
            )
            | "Reshuffle" >> beam.Reshuffle()
            | "Saving Data" >> beam.MapTuple(save_data, directory=output_dir_prefix)
        )
