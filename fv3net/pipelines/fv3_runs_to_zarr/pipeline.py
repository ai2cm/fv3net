import logging
import os
import re
import tempfile
from typing import Iterator

import apache_beam as beam
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions

from vcm.cloud import gsutil
from vcm.convenience import rundir

logging.basicConfig(level=logging.INFO)

bucket = "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/one-step-run/C48"
OUTPUT = "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/zarr_new_dims/C48"
coarsenings = (8, 16, 32, 64)


def time_step(file):
    pattern = re.compile(r"(........\.......)")
    return pattern.search(file).group(1)


def get_time_steps(bucket):
    files = gsutil.list_matches(bucket)
    return [time_step(file) for file in files]


def url(time):
    return os.path.join(bucket, time, "rundir")


def convert_to_zarr(time: str) -> Iterator[xr.Dataset]:
    with tempfile.TemporaryDirectory() as dir:
        gsutil.copy_directory_contents(url(time), dir)
        ds = rundir.rundir_to_dataset(dir, time)
        ds.to_zarr(f"{time}.zarr", mode="w")
        remote_path = f"{OUTPUT}/{time}.zarr"
        gsutil.copy(f"{time}.zarr", remote_path)


def run(beam_options):

    timesteps = get_time_steps(bucket)
    print(f"Processing {len(timesteps)} points")
    with beam.Pipeline(options=beam_options) as p:
        (p | beam.Create(timesteps) | "ConvertToZarr" >> beam.ParDo(convert_to_zarr))


if __name__ == "__main__":
    """Main function"""
    beam_options = PipelineOptions(save_main_session=True)
    run(beam_options=beam_options)
