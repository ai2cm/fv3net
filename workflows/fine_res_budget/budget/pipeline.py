import logging
from itertools import product

import apache_beam as beam
import dask
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.utils import retry

from fv3net.pipelines.common import (
    chunks_1d_to_slices
)

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


def yield_indices(merged, dims):
    logger.info("Yielding Indices")
    indexes = [
        merged[dim].values.tolist()
        for dim in dims
    ]
    for time, tile in product(*indexes):
        yield dict(zip(dims, indexes))


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


class TimeTiles(beam.PTransform):
    def __init__(self, dims):
        self.dims = dims
    def expand(self, merged):
        return (
            merged
            | beam.ParDo(yield_indices, self.dims)
            # reshuffle to ensure that data is distributed to workers
            | "Reshuffle Tiles" >> beam.Reshuffle()
            | "Select Data"
            >> beam.Map(
                lambda index, ds: ds.sel(index), beam.pvalue.AsSingleton(merged)
            )
        )


def yield_time_physics_time_slices(merged):
    # grab a physics variable
    omega = merged['omega']
    time_slices = chunks_1d_to_slices(omega.chunks['time'])

    for slice_ in time_slices:
        for tile in merged.tiles.values.tolist():
            yield {'time': slice_, 'tile': tile}


class OpenTimeChunks(beam.PTransform):
    def expand(self, merged):
        time_tiles = merged | beam.Map(yield_time_physics_time_slices)

        chunk_inds = (
            merged | beam
        )


def run(restart_url, physics_url, output_dir, extra_args=()):

    options = PipelineOptions(extra_args)

    with beam.Pipeline(options=options) as p:

        (
            p
            | FunctionSource(open_merged, restart_url, physics_url)
            | TimeTiles(dims=['time', 'tile'])
            | "Compute Budget" >> beam.Map(budgets.compute_recoarsened_budget, factor=8)
            | "Load" >> beam.Map(load)
            | "Save" >> beam.Map(save, base=output_dir)
        )
