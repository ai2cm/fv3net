import apache_beam as beam
import tempfile
import os
import logging
import gcsfs
from pathlib import Path
import dask

import pandas as pd
import xarray as xr

from apache_beam.io.filesystems import FileSystems

from apache_beam.options.pipeline_options import PipelineOptions

from fv3net.pipelines.common import list_timesteps, FunctionSource
import vcm
from vcm.cloud import gcs
from vcm.coarsen import _save_restart_categories
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


def _coarsen_data(
    local_coarsen_dir, timestep_name, coarsen_factor, grid_spec, local_timestep_dir
):

    os.makedirs(local_coarsen_dir, exist_ok=True)
    logger.debug(f"Directory for coarsening files: {local_coarsen_dir}")
    filename_prefix = f"{timestep_name}."

    from vcm.coarsen import _open_restart_categories

    # open source
    source_data_prefix = os.path.join(
        local_timestep_dir, timestep_name, filename_prefix
    )
    data_pattern = "{prefix}{category}.tile{tile}.nc"
    source = _open_restart_categories(source_data_prefix, data_pattern=data_pattern)

    # import computation
    for category, data in vcm.coarsen_restarts_on_pressure(
        coarsen_factor, grid_spec, source
    ).items():
        yield timestep_name, category, data


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


def coarsen_timestep(
    curr_timestep, coarsen_factor: int, gridspec: xr.Dataset, prefix="."
):

    tmp_timestep_dir = os.path.join(prefix, "local_fine_dir")
    local_coarsen_dir = os.path.join(prefix, "local_coarse_dir", curr_timestep)

    yield from _coarsen_data(
        local_coarsen_dir, curr_timestep, coarsen_factor, gridspec, tmp_timestep_dir,
    )


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
                gridspec=beam.pvalue.AsSingleton(grid_spec),
                # TODO remove this argument
                prefix=prefix,
            )
            | "Saving Data" >> beam.MapTuple(save_data, directory=output_dir_prefix)
        )
