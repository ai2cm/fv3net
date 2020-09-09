import argparse
import fsspec
import json
from typing import Sequence
import xarray as xr
import zarr.storage as zstore

from diagnostics_utils import RegionOfInterest


rename_diag_table_vars = {
    "grid_xt": "x",
    "grid_x": "x_interface",
    "grid_yt": "y",
    "grid_y": "y_interface",
    "pfull": "z",
    "temp": "air_temperature",
    "sphum": "specific_humidity",
}


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()

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
        "--data-paths",
        nargs="*",
        type=str,
        help="Location of data."
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


def _get_source_zarr_datasets(
    urls: Sequence[str], consolidated: bool = False
) -> Sequence[xr.Dataset]:
    datasets = []
    for url in urls:
        mapper = fsspec.get_mapper(url)
        ds = xr.open_zarr(
            zstore.LRUStoreCache(mapper, 1024),
            consolidated=consolidated,
            mask_and_scale=False,
        )
        datasets.append(ds)
    return xr.merge(datasets)
        

if __name__ == "__main__":
    args = _create_arg_parser()
    if all([".zarr" in url for url in args.data_paths]):
        ds = _get_source_zarr_datasets(args.data_paths, args.consolidated)

