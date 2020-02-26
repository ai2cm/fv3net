import apache_beam as beam
import tempfile
import os
import logging
import gcsfs
from pathlib import Path
from apache_beam.options.pipeline_options import PipelineOptions

from vcm.cloud import gcs
from ..common import parse_timestep_from_path, list_timesteps
import vcm

logger = logging.getLogger("CoarsenPipeline")
logger.setLevel(logging.DEBUG)

NUM_FILES_IN_COARSENED_DIR = 24


def check_coarsen_incomplete(gcs_url, output_prefix):

    timestep = parse_timestep_from_path(gcs_url)
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


def _coarsen_data(
    local_coarsen_dir, timestep_name, coarsen_factor, local_spec_dir, local_timestep_dir
):

    os.makedirs(local_coarsen_dir, exist_ok=True)
    logger.debug(f"Directory for coarsening files: {local_coarsen_dir}")
    filename_prefix = f"{timestep_name}."
    vcm.coarsen_restarts_on_pressure(
        coarsen_factor,
        os.path.join(local_spec_dir, "grid_spec"),
        os.path.join(local_timestep_dir, timestep_name, filename_prefix),
        os.path.join(local_coarsen_dir, filename_prefix),
    )
    logger.info("Coarsening completed.")


def _upload_data(gcs_output_dir, local_coarsen_dir):

    logger.info(f"Uploading coarsened data to {gcs_output_dir}")
    bucket_name, blob_prefix = gcs.parse_gcs_url(gcs_output_dir)
    gcs.upload_dir_to_gcs(bucket_name, blob_prefix, Path(local_coarsen_dir))


def coarsen_timestep(
    timestep_gcs_url: str, output_dir: str, coarsen_factor: int, gridspec_path: str
):

    curr_timestep = parse_timestep_from_path(timestep_gcs_url)
    logger.info(f"Coarsening timestep: {curr_timestep}")

    with tempfile.TemporaryDirectory() as tmpdir:

        local_spec_dir = os.path.join(tmpdir, "local_grid_spec")
        _copy_gridspec(local_spec_dir, gridspec_path)

        tmp_timestep_dir = os.path.join(tmpdir, "local_fine_dir")
        _copy_restart(tmp_timestep_dir, timestep_gcs_url)

        local_coarsen_dir = os.path.join(tmpdir, "local_coarse_dir", curr_timestep)
        _coarsen_data(
            local_coarsen_dir,
            curr_timestep,
            coarsen_factor,
            local_spec_dir,
            tmp_timestep_dir,
        )

        timestep_output_dir = os.path.join(output_dir, curr_timestep)
        _upload_data(timestep_output_dir, local_coarsen_dir)


def run(args, pipeline_args=None):

    source_timestep_dir = args.gcs_src_dir
    gridspec_path = args.gcs_grid_spec_path
    source_resolution = args.source_resolution
    target_resolution = args.target_resolution

    if args.gcs_dst_dir:
        output_dir_prefix = args.gcs_dst_dir
        if not args.no_target_subdir:
            output_dir_prefix = os.path.join(output_dir_prefix, f"C{target_resolution}")
    else:
        output_dir_prefix = os.path.join(source_timestep_dir, f"C{target_resolution}")

    coarsen_factor = source_resolution // target_resolution
    available_timesteps = list_timesteps(source_timestep_dir)
    timestep_urls = [
        os.path.join(source_timestep_dir, tstep) for tstep in available_timesteps
    ]

    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | "CreateTStepURLs" >> beam.Create(timestep_urls)
            | "CheckCompleteTSteps"
            >> beam.Filter(check_coarsen_incomplete, output_dir_prefix)
            | "CoarsenTStep"
            >> beam.ParDo(
                coarsen_timestep,
                output_dir=output_dir_prefix,
                coarsen_factor=coarsen_factor,
                gridspec_path=gridspec_path,
            )
        )
