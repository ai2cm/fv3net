import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import re
import tempfile
import os
import logging

from vcm.cloud import gsutil
import vcm

logger = logging.getLogger('CoarsenTimesteps')
logger.setLevel(logging.INFO)


def time_step(file):
    pattern = re.compile(r"(........\.......)")
    return pattern.search(file).group(1)


def coarsen_timestep(
    timestep_gcs_url: str,
    output_dir: str,
    coarsen_factor: int,
    gridspec_path: str,
):

    curr_timestep = time_step(timestep_gcs_url)
    logger.info('Coarsening timestep')

    # Download data restart, spec
    with tempfile.TemporaryDirectory as tmpdir:

        # Copy Gridspec
        local_spec_dir = os.path.join(tmpdir, 'local_grid_spec')
        os.makedirs(local_spec_dir, exist_ok=True)
        gsutil.copy(gridspec_path, local_spec_dir)

        # Copy restart
        tmp_timestep_dir = os.path.join(tmpdir, time_step)
        gsutil.copy(timestep_gcs_url, tmp_timestep_dir)

        # Coarsen data
        local_coarse_dir = os.path.join(tmpdir, 'local_coarse_dir', curr_timestep)
        os.makedirs(local_coarse_dir, exist_ok=True)
        vcm.coarsen_restarts_on_pressure(
            coarsen_factor,
            os.path.join(local_spec_dir, 'grid_spec'),
            tmp_timestep_dir,
            local_coarse_dir
        )

        # Upload data
        timstep_gcs_output_dir = os.path.join(output_dir, curr_timestep)
        gsutil.copy(local_coarse_dir, timstep_gcs_output_dir)


def run(args, pipeline_args=None):

    source_timestep_dir = args.gcs_src_dir
    gridspec_path = args.gcs_grid_spec_path
    source_resolution = args.source_resolution
    target_resolution = args.target_resolution
    
    if args.gcs_dst_dir:
        output_dir_prefix = os.path.join(args.gcs_dst_dir, target_resolution)
    else:
        output_dir_prefix = os.path.join(source_timestep_dir, target_resolution)

    coarsen_factor = source_resolution // target_resolution
    timestep_urls = gsutil.list_matches(source_timestep_dir)
    timestep_urls = timestep_urls[0:2]
    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | "CreateTStepURLs" >> beam.Create(timestep_urls)
            | "CoarsenTStep" >> beam.ParDo(coarsen_timestep,
                                           output_dir=output_dir_prefix,
                                           coarsen_factor=coarsen_factor,
                                           gridspec_path=gridspec_path)
        )
