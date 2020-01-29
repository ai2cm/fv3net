import apache_beam as beam
import re
import tempfile
import os
import logging
import gcsfs
from pathlib import Path
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud.storage import Client

from vcm.cloud import gcs
import vcm

logger = logging.getLogger("CoarsenPipeline")
logger.setLevel(logging.DEBUG)

NUM_FILES_IN_COARSENED_DIR = 24


def time_step(file):
    pattern = re.compile(r"(........\.......)")
    return pattern.search(file).group(1)


def download_all_bucket_files(
    gcs_url: str, out_dir_prefix: str, include_parent_in_stem=True
):
    bucket_name, blob_prefix = gcs.parse_gcs_url(gcs_url)
    blob_gcs_paths = gcs.list_bucket_files(Client(), bucket_name, prefix=blob_prefix)
    parent_dirname = Path(blob_prefix).name
    logger.debug(f"Downloading files from bucket prefix: {blob_prefix}")

    for blob_url in blob_gcs_paths:
        _, blob_path = gcs.parse_gcs_url(blob_url)
        filename = Path(blob_path).name
        full_dir = str(Path(blob_path).parent)
        out_dir_stem = _get_dir_stem(
            parent_dirname, full_dir, include_parent=include_parent_in_stem
        )

        blob = gcs.init_blob_from_gcs_url(blob_url)
        out_dir = os.path.join(out_dir_prefix, out_dir_stem)
        gcs.download_blob_to_file(blob, out_dir, filename)


def _get_dir_stem(parent_dirname, full_dirname, include_parent=True):

    dir_components = Path(full_dirname).parts
    stem_start_idx = dir_components.index(parent_dirname)

    if not include_parent:
        stem_start_idx += 1

    stem_dir = dir_components[stem_start_idx:]

    if stem_dir:
        stem_dir = str(Path(*stem_dir))
    else:
        stem_dir = ""

    return stem_dir


def check_coarsen_incomplete(gcs_url, output_prefix):

    output_timestep_dir = os.path.join(output_prefix, time_step(gcs_url))

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
    download_all_bucket_files(
        gridspec_gcs_path, local_spec_dir, include_parent_in_stem=False
    )


def _copy_restart(local_timestep_dir, timestep_gcs_path):

    os.makedirs(local_timestep_dir, exist_ok=True)
    logger.debug(f"Copying restart files to {local_timestep_dir}")
    download_all_bucket_files(timestep_gcs_path, local_timestep_dir)


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
    timestep_gcs_url: str, output_dir: str, coarsen_factor: int, gridspec_path: str,
):

    curr_timestep = time_step(timestep_gcs_url)
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
        output_dir_prefix = os.path.join(args.gcs_dst_dir, f"C{target_resolution}")
    else:
        output_dir_prefix = os.path.join(source_timestep_dir, f"C{target_resolution}")

    coarsen_factor = source_resolution // target_resolution
    fs = gcsfs.GCSFileSystem()
    timestep_urls = fs.ls(source_timestep_dir)

    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | "CreateTStepURLs" >> beam.Create(timestep_urls)
            | "CheckCompleteTSteps"
            >> beam.filter(check_coarsen_incomplete, output_dir_prefix)
            | "CoarsenTStep"
            >> beam.ParDo(
                coarsen_timestep,
                output_dir=output_dir_prefix,
                coarsen_factor=coarsen_factor,
                gridspec_path=gridspec_path,
            )
        )
