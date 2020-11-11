import socket
import datetime
import sys
import logging
from typing import Iterable, Sequence, Hashable, Mapping
from itertools import product
import os
import xarray as xr

import apache_beam as beam
import dask
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.utils import retry

from fv3net.pipelines.common import FunctionSource

import vcm

from . import config
from . import budgets
from .data import open_merged

dask.config.set(scheduler="single-threaded")


logger = logging.getLogger(__file__)


Dims = Sequence[Hashable]


def yield_indices(merged: xr.Dataset, dims: Dims) -> Iterable[Mapping[Hashable, slice]]:
    logger.info("Yielding Indices")
    indexes = [merged[dim].values.tolist() for dim in dims]  # type: ignore
    for index in product(*indexes):
        yield dict(zip(dims, index))


@retry.with_exponential_backoff(num_retries=7)
def save(ds: xr.Dataset, base: str):
    time = vcm.encode_time(ds.time.item())
    tile = ds.tile.item()

    path = os.path.join(base, f"{time}.tile{tile}.nc")
    logger.info(f"saving data to {path}")

    try:
        FileSystems.mkdirs(base)
    except IOError:
        pass

    with FileSystems.create(path) as f:
        vcm.dump_nc(ds, f)


class OpenTimeChunks(beam.PTransform):
    def expand(self, merged):
        return (
            merged
            | "YieldPhysicsChunk" >> beam.ParDo(yield_indices, ["time", "tile"])
            | beam.Reshuffle()
            | beam.Map(lambda index, ds: ds.sel(index), beam.pvalue.AsSingleton(merged))
        )


def run(restart_url, physics_url, gfsphysics_url, output_dir, extra_args=()):

    options = PipelineOptions(extra_args)
    with beam.Pipeline(options=options) as p:

        (
            p
            | FunctionSource(open_merged, restart_url, physics_url, gfsphysics_url)
            | OpenTimeChunks()
            | "Compute Budget"
            >> beam.Map(
                budgets.compute_recoarsened_budget_inputs,
                factor=config.factor,
                first_moments=config.VARIABLES_TO_AVERAGE,
            )
            | "Insert metadata"
            >> beam.Map(
                lambda x: x.assign_attrs(
                    history=" ".join(sys.argv),
                    restart_url=restart_url,
                    physics_url=physics_url,
                    gfsphysics_url=gfsphysics_url,
                    launching_host=socket.gethostname(),
                    date_computed=datetime.datetime.now().isoformat(),
                )
            )
            | "Save" >> beam.Map(save, base=output_dir)
        )
