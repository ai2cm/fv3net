import logging
from typing import Iterable, Sequence, Hashable, Mapping
from itertools import product
import tempfile
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


def chunks_1d_to_slices(chunks: Iterable[int]) -> Iterable[slice]:
    start = 0
    for chunk in chunks:
        end = start + chunk
        yield slice(start, end)
        start = end


def yield_indices(merged: xr.Dataset, dims: Dims) -> Iterable[Mapping[Hashable, slice]]:
    logger.info("Yielding Indices")
    indexes = [merged[dim].values.tolist() for dim in dims]  # type: ignore
    for index in product(*indexes):
        yield dict(zip(dims, index))


def split_by_dim(merged, dims: Dims) -> Iterable[xr.Dataset]:
    for index in yield_indices(merged, dims):
        logger.info(f"split_by_dim: {index}")
        yield merged.sel(index)


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


@retry.with_exponential_backoff(num_retries=7)
def load(ds: xr.Dataset) -> xr.Dataset:
    logger.info(f"Loading {ds}")
    return ds.compute()


def yield_time_physics_time_slices(merged: xr.Dataset) -> Iterable[Mapping[str, slice]]:
    # grab a physics variable
    omega = merged["vulcan_omega_coarse"]
    chunks = omega.chunks[omega.get_axis_num("time")]
    time_slices = chunks_1d_to_slices(chunks)

    for slice_ in time_slices:
        for tile in range(len(merged.tile)):
            yield {"time": slice_, "tile": slice(tile, tile + 1)}


def _clear_encoding(ds):
    ds.encoding = {}
    for variable in ds:
        ds[variable].encoding = {}


def _load_and_split(ds: xr.Dataset, dims: Sequence[str]) -> Iterable[xr.Dataset]:
    # cache on disk to avoid OOM
    with tempfile.TemporaryDirectory() as dir_:
        path = os.path.join(dir_, "data.zarr")
        chunks = dict(zip(dims, [1] * len(dims)))
        rechunked = ds.chunk(chunks=chunks)  # type: ignore
        _clear_encoding(rechunked)
        logger.info("Saving data to disk")

        # Save variable by variable to avoid OOM
        for variable in rechunked:
            logger.info(f"Writing {variable} to disk")
            rechunked[variable].to_dataset().to_zarr(path, mode="a")

        # open data
        ds = xr.open_zarr(path)
        for data in split_by_dim(ds, dims):
            logger.info(f"Yielding {data}")
            yield data.load()


class OpenTimeChunks(beam.PTransform):
    def expand(self, merged):
        return (
            merged
            | "YieldPhysicsChunk" >> beam.ParDo(yield_time_physics_time_slices)
            | beam.Reshuffle()
            | beam.Map(
                lambda index, ds: ds.isel(index), beam.pvalue.AsSingleton(merged)
            )
            | "LoadData" >> beam.ParDo(_load_and_split, ["time", "tile"])
        )


def run(restart_url, physics_url, atmos_avg_url, output_dir, extra_args=()):

    options = PipelineOptions(extra_args)
    with beam.Pipeline(options=options) as p:

        (
            p
            | FunctionSource(open_merged, restart_url, physics_url, atmos_avg_url)
            | OpenTimeChunks()
            | "Compute Budget"
            >> beam.Map(
                budgets.compute_recoarsened_budget_inputs,
                factor=config.factor,
                first_moments=config.VARIABLES_TO_AVERAGE,
            )
            | "Load" >> beam.Map(load)
            | "Save" >> beam.Map(save, base=output_dir)
        )
