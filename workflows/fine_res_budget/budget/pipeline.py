import logging
from itertools import product

import apache_beam as beam

import dask
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.utils import retry

import vcm
from vcm import safe

from . import budgets
from .data import merge, open_diagnostic_output, open_restart_data

dask.config.set(scheduler="single-threaded")


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
        yield merged.sel(time=time).isel(tile=tile).assign_coords(tile=tile)


@retry.with_exponential_backoff(num_retries=7)
def save(ds, base):
    time = vcm.encode_time(ds.time.item())
    tile = ds.tile.item() + 1

    path = f"{base}/{time}.tile{tile}.nc"
    logger.info(f"saving data to {path}")

    try:
        FileSystems.mkdirs(base)
    except IOError:
        pass

    with FileSystems.create(path) as f:
        vcm.dump_nc(ds, f)


@retry.with_exponential_backoff(num_retries=7)
def load(ds):
    return ds.compute()


def run(restart_url, physics_url, output_dir, extra_args=()):

    options = PipelineOptions(extra_args)

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
