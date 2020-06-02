import argparse
import logging
import os
from typing import Iterable, Mapping, Tuple, TypeVar

import apache_beam as beam
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions

import vcm
from fv3net.pipelines.common import FunctionSource, WriteToNetCDFs, list_timesteps

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
            # Data is still lazy at this point so reshuffle is not too costly
            # It distributes the work
            | "Reshuffle" >> beam.Reshuffle()
            | "OpenRestartLazy" >> beam.Map(_open_restart_categories, src_dir)
            | "CoarsenTStep"
            >> beam.ParDo(
                coarsen_timestep,
                coarsen_factor=factor,
                grid_spec=beam.pvalue.AsSingleton(grid_spec),
            )
            # Reduce problem size by splitting by tiles
            # This will result in repeated reads to each fv_core file
            | "Split By Tiles" >> beam.ParDo(split_by_tiles)
            | "Force evaluation" >> beam.Map(load)
            | WriteToNetCDFs(output_filename, output_dir)
        )


def main(argv):

    logger = logging.getLogger("CoarsenTimesteps")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "src_dir", type=str, help="Full path to input data for downloading timesteps.",
    )
    parser.add_argument(
        "grid_spec_path",
        type=str,
        help=(
            "Full path to directory containing files matching 'grid_spec.tile*.nc' to "
            "select grid specfiles with same resolution as the source data."
        ),
    )
    parser.add_argument(
        "source_resolution", type=int, help="Source data cubed-sphere grid resolution.",
    )
    parser.add_argument(
        "target_resolution", type=int, help="Target coarsening resolution to output.",
    )
    parser.add_argument(
        "dst_dir",
        type=str,
        help=(
            "Full path to output coarsened timestep data. Defaults to input path"
            "with target resolution appended as a directory"
        ),
    )
    parser.add_argument(
        "--add-target-subdir",
        action="store_true",
        help=(
            "Add subdirectory with C{target-resolution} to the specified "
            "destination directory. "
        ),
    )
    logging.basicConfig(level=logging.INFO)

    args, pipeline_args = parser.parse_known_args(argv)

    gridspec_path = os.path.join(args.grid_spec_path, "grid_spec")

    output_dir_prefix = args.dst_dir
    if args.add_target_subdir:
        output_dir_prefix = os.path.join(
            output_dir_prefix, f"C{args.target_resolution}"
        )

    factor = args.source_resolution // args.target_resolution

    run(
        gridspec_path,
        args.src_dir,
        output_dir_prefix,
        factor,
        pipeline_args=pipeline_args,
    )
