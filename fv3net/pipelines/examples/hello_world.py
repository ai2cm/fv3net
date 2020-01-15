import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from vcm.cloud import gsutil

logging.basicConfig(level=logging.INFO)


def gslist(i):
    logging.info(gsutil.list_matches("gs://vcm-ml-data/"))


def run(beam_options):
    timesteps = [1]
    print(f"Processing {len(timesteps)} points")
    with beam.Pipeline(options=beam_options) as p:
        (p | beam.Create(timesteps) | "ConvertToZarr" >> beam.ParDo(gslist))


if __name__ == "__main__":
    """Main function"""
    beam_options = PipelineOptions(save_main_session=True)
    run(beam_options=beam_options)
