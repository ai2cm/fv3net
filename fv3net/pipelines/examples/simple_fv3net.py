import apache_beam as beam
import os

from apache_beam.options.pipeline_options import PipelineOptions

import fv3net
import vcm
import mappm
import fv3config


def run(list_to_sum, out_gcs_dir, pipeline_options=None):

    out_gcs_path = os.path.join(out_gcs_dir, "SUMFILE.txt")
    options = PipelineOptions(flags=pipeline_options)
    with beam.Pipeline(options=options) as p:
        (
            p
            | "CreateCollection" >> beam.Create(list_to_sum)
            | "SummedNumbers" >> beam.CombineGlobally(sum)
            | "WriteSum" >> beam.io.WriteToText(out_gcs_path, shard_name_template='', num_shards=1)
        )
