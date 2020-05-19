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


def open_merged(restart_url, physics_url):
    restarts = open_restart_data(restart_url)
    diag = open_diagnostic_output(physics_url)
    data = safe.get_variables(merge(restarts, diag), PHYSICS_VARIABLES)
    num_tiles = len(data.tile)
    tiles = range(1, num_tiles + 1)
    return data.assign_coords(tile=tiles)


def yield_indices(merged):
    logger.info("Yielding Indices")
    times = merged["time"].values.tolist()
    tiles = merged["tile"].values.tolist()
    for time, tile in product(times, tiles):
        yield {"time": time, "tile": tile}


@retry.with_exponential_backoff(num_retries=7)
def save(ds, base):
    time = vcm.encode_time(ds.time.item())
    tile = ds.tile.item()

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
    logger.info(f"Loading {ds}")
    return ds.compute()


class FunctionSource(beam.PTransform):
    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args

    def expand(self, pcoll):
        return pcoll | beam.Create([None]) | beam.Map(lambda _: self.fn(*self.args))


class ChunkByTimeTile(beam.PTransform):
    def expand(self, merged):
        return (
            merged
            | beam.ParDo(yield_indices)
            # reshuffle to ensure that data is distributed to workers
            | "Reshuffle Tiles" >> beam.Reshuffle()
            | "Select Data"
            >> beam.Map(
                lambda index, ds: ds.sel(index), beam.pvalue.AsSingleton(merged)
            )
        )


def run(restart_url, physics_url, output_dir, extra_args=()):

    options = PipelineOptions(extra_args)

    with beam.Pipeline(options=options) as p:

        (
            p
            | FunctionSource(open_merged, restart_url, physics_url)
            | ChunkByTimeTile()
            | "Compute Budget" >> beam.Map(budgets.compute_recoarsened_budget, factor=8)
            | "Load" >> beam.Map(load)
            | "Save" >> beam.Map(save, base=output_dir)
        )
