import apache_beam
import dask
import fsspec
import xpartition  # noqa: E402 F401
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.utils import retry
from budget import budgets, config
from budget.data import open_merged

from fv3net.pipelines.common import FunctionSource

dask.config.set(scheduler="single-threaded")


@retry.with_exponential_backoff(num_retries=7)
def _write(rank, mapper):
    return mapper.write(rank)


class WriteAll(apache_beam.PTransform):
    def expand(self, mapper):
        return (
            mapper
            | apache_beam.ParDo(iter)
            | apache_beam.Reshuffle()
            | apache_beam.Map(_write, mapper=apache_beam.pvalue.AsSingleton(mapper))
        )


def func(iData):
    return budgets.compute_recoarsened_budget_inputs(
        iData, config.factor, first_moments=config.VARIABLES_TO_AVERAGE
    ).drop(["step"])


def open_partitioner(restart_url, physics_url, gfsphysics_url, area_url, output_dir):
    data = open_merged(restart_url, physics_url, gfsphysics_url, area_url)
    dims = ["time", "tile"]
    ranks = len(data.time) * len(data.tile)
    mapper = data.delp.isel(step=0).partition.map(
        fsspec.get_mapper(output_dir), ranks, dims, func, data
    )
    return mapper


def run(restart_url, physics_url, gfsphysics_url, area_url, output_dir, extra_args=()):

    options = PipelineOptions(extra_args)
    with apache_beam.Pipeline(options=options) as p:
        (
            p
            | FunctionSource(
                open_partitioner,
                restart_url,
                physics_url,
                gfsphysics_url,
                area_url,
                output_dir,
            )
            | WriteAll()
        )
