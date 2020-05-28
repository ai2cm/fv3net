import apache_beam as beam
import os
import logging
from typing import Mapping, Iterable, Tuple, List, TypeVar

import xarray as xr


from apache_beam.options.pipeline_options import PipelineOptions

from fv3net.pipelines.common import list_timesteps, FunctionSource, WriteToNetCDFs
import vcm
from vcm import parse_timestep_str_from_path

logger = logging.getLogger("CoarsenPipeline")
logger.setLevel(logging.DEBUG)


def output_filename(arg: Tuple[str, str, int], directory: str) -> str:
    time, category, tile = arg
    assert tile in [1, 2, 3, 4, 5, 6]
    return os.path.join(directory, time, f"{time}.{category}.tile{tile}.nc")


def _open_restart_categories(
    time: str, prefix: str
) -> Tuple[str, Mapping[str, xr.Dataset]]:
    source = {}

    OUTPUT_CATEGORY_NAMES = {
        "fv_core.res": "fv_core_coarse.res",
        "fv_srf_wnd.res": "fv_srf_wnd_coarse.res",
        "fv_tracer.res": "fv_tracer_coarse.res",
        "sfc_data": "sfc_data_coarse",
    }

    for output_category in OUTPUT_CATEGORY_NAMES:
        input_category = OUTPUT_CATEGORY_NAMES[output_category]
        tile_prefix = os.path.join(prefix, time, f"{time}.{input_category}")
        source[output_category] = vcm.open_tiles(tile_prefix)
    return time, source


def coarsen_timestep(
    arg: Tuple[str, Mapping[str, xr.Dataset]],
    coarsen_factor: int,
    grid_spec: xr.Dataset,
) -> Iterable[Tuple[Tuple[str, str], Mapping[str, xr.Dataset]]]:

    time, source = arg
    for category, data in vcm.coarsen_restarts_on_pressure(
        coarsen_factor, grid_spec, source
    ).items():
        yield (time, category), data


def split_by_tiles(kv: Tuple[Tuple, xr.Dataset]) -> Iterable[Tuple[Tuple, xr.Dataset]]:
    key, ds = kv
    for tile in range(6):
        yield key + (tile + 1,), ds.isel(tile=tile)


T = TypeVar("T")


def load(kv: Tuple[T, xr.Dataset]) -> Tuple[T, xr.Dataset]:
    key, val = kv
    return key, val.load()


def run(
    gridspec_path: str, src_dir: str, output_dir: str, factor: int, pipeline_args=None,
):

    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)

    with beam.Pipeline(options=beam_options) as p:
        grid_spec = p | FunctionSource(
            lambda x: vcm.open_tiles(x).load(), gridspec_path
        )
        (
            p
            | beam.Create([src_dir]).with_output_types(str)
            | "ListTimes" >> beam.ParDo(list_timesteps)
            | "ParseTimeString" >> beam.Map(vcm.parse_timestep_str_from_path)
            | "Reshuffle 1" >> beam.Reshuffle()
            | "OpenRestartLazy" >> beam.Map(_open_restart_categories, src_dir)
            | "CoarsenTStep"
            >> beam.ParDo(
                coarsen_timestep,
                coarsen_factor=factor,
                grid_spec=beam.pvalue.AsSingleton(grid_spec),
            )
            # Reduce problem size by splitting by tiles
            # This will results in repeated reads to each fv_core file
            | "Split By Tiles" >> beam.ParDo(split_by_tiles)
            # Data is still lazy at this point so reshuffle is not too costly
            # It distributes the work
            | "Reshuffle" >> beam.Reshuffle()
            | "Force evaluation" >> beam.Map(load)
            | WriteToNetCDFs(output_filename, output_dir)
        )
