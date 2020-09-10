import argparse
from datetime import datetime
import fsspec
import json
from typing import Sequence
import xarray as xr
import zarr.storage as zstore

from diagnostics_utils import RegionOfInterest
import ._utils as utils

TIME_FMT = "%Y%m%d.%H%M%S"

rename_vars = {
    "grid_xt": "x",
    "grid_x": "x_interface",
    "grid_yt": "y",
    "grid_y": "y_interface",
    "pfull": "z",
    "delp": "pressure_thickness_of_atmospheric_layer",
    "temp": "air_temperature",
    "sphum": "specific_humidity",
    "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
    "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
}


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_data_path",
        type=str,
        help="Location of run data."
    )
    parser.add_argument(
        "fine_res_reference_path",
        type=str,
        help="Location of reference fine res data."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help=("Local or remote path where diagnostic dataset will be written."),
    )
    parser.add_argument(
        "lat_bounds",
        nargs=2,
        type=float,
        help=(
            "Min, max latitude bounds"
        ),
    )
    parser.add_argument(
        "lon_bounds",
        nargs=2,
        type=float,
        help=(
            "Min, max longitude bounds"
        ),
    )

    parser.add_argument(
        "--consolidated",
        type=bool,
        default=False,
        help="Is zarr metadata consolidated?"
    )
    parser.add_argument(
        "--time-bounds",
        nargs=2,
        type=str,
        help="Optional, min/max time range. Should have format 'YYYYMMDD.HHMMSS'."
    )
    parser.add_argument(
        "--mapper-function",
        type=str,
        help="Optional, provide if reading vertical profiles from training data."
    )
    parser.add_argument(
        "--mapper-kwargs",
        type=json.loads,
        help="Optional, use if using a mapper to read training data."
    )
    return parser.parse_args()


def _open_zarr(
    url: str, time_bounds: Sequence[str], consolidated: bool = False
) :
    mapper = fsspec.get_mapper(url)
    time_slice = slice(*[datetime.strptime(t, TIME_FMT) for t in time_bounds])
    ds = xr.open_zarr(
        zstore.LRUStoreCache(mapper, 1024),
        consolidated=consolidated,
        mask_and_scale=False,
    ) 
    renamed = {
        key: value for key, value in rename_vars.items()
        if key in ds.data_vars}
    ds = ds.rename(renamed) \
        .pipe(utils.standardize_zarr_time_coord) \
        .sel({"time": time_slice})
    return ds


def _fine_res_reference(
    fine_res_path: str,
    times: Sequence[datetime.datetime]
):
    mapper = open_fine_res_apparent_sources(fine_res_path)
    times = [t.strftime(TIME_FMT) for t in times]
    time_slice = slice(*[datetime.strptime(t, TIME_FMT) for t in time_bounds])
    return utils.dataset_from_timesteps(
        mapper, times, ["air_temperature", "specific_humidity"])


if __name__ == "__main__":
    args = _create_arg_parser()
    if ".zarr" in args.data_path:
        ds = _open_zarr(args.data_paths, args.time_bounds, args.consolidated,)

    fine_res = _fine_res_reference(args.fine_res_reference_path, ds.time.values)
    