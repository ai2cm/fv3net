import apache_beam as beam
from src import gcs
from .core import coarsen_and_upload_surface, output_names
from apache_beam.options.pipeline_options import PipelineOptions  
import logging
import re
from apache_beam.utils import retry
logging.basicConfig(level=logging.INFO)

bucket = 'gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/C3702/'
coarsenings = (8, 16, 32, 64)

def time_step(file):
    pattern = re.compile(r'(........\.......)')
    return pattern.search(file).group(1)


def get_completed_time_steps():
    files = gcs.list_matches(bucket)
    return [(time_step(file), coarsenings) for file in files]


def is_not_done(key):
    return any(not gcs.exists(url) for url in output_names(key).values())
     

def run(beam_options):
    timesteps = get_completed_time_steps()
    print(f"Processing {len(timesteps)} points")
    coarse_fn = retry.with_exponential_backoff(initial_delay_secs=30)(coarsen_and_upload_surface)
    with beam.Pipeline(options=beam_options) as p:
        (p | beam.Create(timesteps)
           | beam.Filter(is_not_done)
           | 'DownloadCoarsen' >> beam.ParDo(coarse_fn)
           )

if __name__ == '__main__':
  """Main function"""
  beam_options = PipelineOptions(save_main_session=True)
  run(beam_options=beam_options)
