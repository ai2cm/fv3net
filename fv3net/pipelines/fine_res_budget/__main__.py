import apache_beam as beam
import joblib
import xarray as xr
import datetime
import os
from itertools import product
import vcm
from vcm import safe
from .data import open_restart_data, open_diagnostic_output, merge
from apache_beam.utils import retry
from apache_beam.io.filesystems import FileSystems

from apache_beam.options.pipeline_options import PipelineOptions

import argparse

# from vcm import safe
import dask
from . import budgets

dask.config.set(scheduler="single-threaded")

import logging

logger = logging.getLogger(__file__)


PHYSICS_VARIABLES = [
    "omega",
    "t_dt_gfdlmp",
    "t_dt_nudge",
    "t_dt_phys",
    "qv_dt_gfdlmp",
    "qv_dt_phys",
    "eddy_flux_omega_sphum",
    "eddy_flux_omega_temp",
    "omega",
    "delp",
    "sphum",
    "T",
    "area",
]


def open_merged(_, restart_url, physics_url):
    restarts = open_restart_data(restart_url)
    diag = open_diagnostic_output(physics_url)
    return safe.get_variables(merge(restarts, diag), PHYSICS_VARIABLES)


def yield_all(merged):
    times = merged.time.values.tolist()
    tiles = merged.tile.values.tolist()
    for time, tile in product(times, tiles):
        yield merged.sel(time=time, tile=tile)


@retry.with_exponential_backoff(num_retries=7)
def save(ds, base):
    time = vcm.encode_time(ds.time.item())
    tile = ds.tile.item() + 1

    path = f"{base}/{time}.tile{tile}.nc"
    logger.info(f"saving data to {path}")
    with FileSystems.create(path) as f:
        vcm.dump_nc(ds, f)


@retry.with_exponential_backoff(num_retries=7)
def load(ds):
    return ds.load()


def run(restart_url, physics_url, output_dir, options=PipelineOptions([])):

    with beam.Pipeline(options=options) as p:
        merged = (
            p
            | beam.Create([None])
            | "OpenMerged" >> beam.Map(open_merged, restart_url, physics_url)
        )

        (
            merged
            | "Yield all tiles" >> beam.ParDo(yield_all)
            | "Compute Budget" >> beam.Map(budgets.compute_recoarsened_budget, factor=8)
            | "Load" >> beam.Map(load)
            | "Save" >> beam.Map(save, base=output_dir)
        )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("physics_url")
    parser.add_argument("restart_url")
    parser.add_argument("output_dir")

    args, extra_args = parser.parse_known_args()
    options = PipelineOptions(extra_args)

    run(args.restart_url, args.physics_url, args.output_dir, options=options)
