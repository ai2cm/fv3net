import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from vcm.cloud import gsutil

logging.basicConfig(level=logging.INFO)


def download_big(i):
    gsutil.copy(i, "local")
    logging.info(gsutil.list_matches("gs://vcm-ml-data/"))


def run(beam_options):
    timesteps = [1]
    print(f"Processing {len(timesteps)} points")
    big_file = (
        "gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-diagnostics/"
        "gfsphysics_15min_fine.tile1.nc.0000"
    )
    with beam.Pipeline(options=beam_options) as p:
        (p | beam.Create([big_file]) | "ConvertToZarr" >> beam.ParDo(download_big))


if __name__ == "__main__":
    """Main function"""
    beam_options = PipelineOptions(save_main_session=True)
    run(beam_options=beam_options)
