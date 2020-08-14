import apache_beam as beam
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
import numpy as np
import subprocess
import logging

import vcm

# some fv3net.pipelines imports
from fv3net.pipelines.common import list_timesteps  # noqa: F401

# a relative import
from .common import FunctionSource  # noqa: F401


def get_package_info():
    return subprocess.check_output(["pip", "freeze"]).decode("UTF-8")


def fake_data(_):
    # simple import to make sure it work
    logging.info("Package Info " + get_package_info())
    vcm.net_heating
    return xr.Dataset({"a": (["x"], np.ones(10))})


logging.basicConfig(level=logging.INFO)

with beam.Pipeline(options=PipelineOptions()) as p:
    (
        p
        | beam.Create([None])
        | "Make Data" >> beam.Map(fake_data)
        | "Reduce" >> beam.Map(lambda x: x.mean())
    )
