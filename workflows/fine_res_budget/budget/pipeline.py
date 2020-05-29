import logging
from typing import Iterable, Sequence, Hashable, Mapping, Any
from itertools import product
import tempfile
import os
import xarray as xr

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

Dims = Sequence[Hashable]


def chunks_1d_to_slices(chunks: Iterable[int]) -> Iterable[slice]:
    start = 0
    for chunk in chunks:
        end = start + chunk
        yield slice(start, end)
        start = end


def open_merged(restart_url: str, physics_url: str) -> xr.Dataset:
    restarts = open_restart_data(restart_url)
    diag = open_diagnostic_output(physics_url)
    data = safe.get_variables(merge(restarts, diag), PHYSICS_VARIABLES)
    num_tiles = len(data.tile)
    tiles = range(1, num_tiles + 1)
    return data.assign_coords(tile=tiles)


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


@beam.typehints.with_input_types(Any)
@beam.typehints.with_output_types(xr.Dataset)
class FunctionSource(beam.PTransform):
    """Create a pcollection by evaluating a single function

    Careful profiling_ indicates that xarray datasets should be loaded within
    a beam ParDo rather than loaded outside the beam graph and passed in by pickling.

    For example, the following naive code is painfully slow::

        ds: Dataset = load_dataset(url)
        p = Pipeline()
        p | beam.Create([ds])
        p.run()

    This equivalent code is about 100 times faster::

        p = Pipeline()
        p | beam.Create([None]) | beam.Map(lambda _, url : load_dataset(url), url)

    This PTransform implements this pattern

    .. _profiling: https://gist.github.com/nbren12/948ef9a5248c43537bb50db49c8851f9

    """

    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args

    def expand(self, pcoll):
        return (
            pcoll
            | beam.Create([None]).with_output_types(None)
            | beam.Map(lambda _: self.fn(*self.args)).with_output_types(xr.Dataset)
        )


def yield_time_physics_time_slices(merged: xr.Dataset) -> Iterable[Mapping[str, slice]]:
    # grab a physics variable
    omega = merged["omega"]
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


def run(restart_url, physics_url, output_dir, extra_args=()):

    options = PipelineOptions(extra_args)
    with beam.Pipeline(options=options) as p:

        (
            p
            | FunctionSource(open_merged, restart_url, physics_url)
            | OpenTimeChunks()
            | "Compute Budget" >> beam.Map(budgets.compute_recoarsened_budget, factor=8)
            | "Load" >> beam.Map(load)
            | "Save" >> beam.Map(save, base=output_dir)
        )
