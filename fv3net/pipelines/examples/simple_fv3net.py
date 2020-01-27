import apache_beam as beam
import os
import tempfile
import subprocess

from apache_beam.options.pipeline_options import PipelineOptions

import fv3net    # noqa
import vcm       # noqa
import mappm     # noqa
import fv3config  # noqa
from vcm.cloud import gsutil

OUTPUT_FILENAME = "SUMFILE.txt"


def run(list_to_sum, out_gcs_dir, pipeline_options=None):

    out_gcs_path = os.path.join(out_gcs_dir, OUTPUT_FILENAME)
    options = PipelineOptions(flags=pipeline_options)
    with beam.Pipeline(options=options) as p:
        (
            p
            | "CreateCollection" >> beam.Create(list_to_sum)
            | "SummedNumbers" >> beam.CombineGlobally(sum)
            | "WriteSum" >> beam.io.WriteToText(out_gcs_path, shard_name_template='', num_shards=1)
        )


def check_run(out_gcs_dir, target_sum):

    dataflow_success_file = os.path.join(out_gcs_dir, OUTPUT_FILENAME)
    assert gsutil.exists(dataflow_success_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_success_file = os.path.join(tmpdir, OUTPUT_FILENAME)
        gsutil.copy(dataflow_success_file, local_success_file)
        with open(local_success_file, 'r') as f:
            sum_result = int(f.readline().strip())
            assert sum_result == target_sum

    subprocess.check_call(['gsutil', 'rm', dataflow_success_file])
