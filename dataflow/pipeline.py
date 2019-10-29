import apache_beam as beam
from src.data.coarsen_surface_data import coarsen_and_upload_surface
from apache_beam.options.pipeline_options import PipelineOptions  
import logging
logging.basicConfig(level=logging.INFO)

timestep = '20160805.064500'

timesteps = [timestep]


def run(beam_options):

    with beam.Pipeline(options=beam_options) as p:
        (p | beam.Create(timesteps)
           | 'DownloadCoarsen' >> beam.ParDo(coarsen_and_upload_surface))


if __name__ == '__main__':
  """Main function"""
  import argparse
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  args, pipeline_args = parser.parse_known_args()
  beam_options = PipelineOptions(pipeline_args, save_main_session=True)
  run(beam_options=beam_options)
