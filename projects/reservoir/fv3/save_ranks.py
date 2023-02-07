import argparse
import fsspec
import intake
import os

from fv3fit.reservoir import CubedsphereRankDivider
import vcm


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_path", type=str, default=None, help=("Path with training data zarr."),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Local or remote path where output will be written.",
    )
    parser.add_argument(
        "layout_width",
        type=int,
        default=1,
        help=(
            "Width of rank layout within each tile, ex. layout_width=2 corresponds "
            "to (2,2) layout with four ranks per tile. Defaults to 1 rank per tile."
        ),
    )
    parser.add_argument(
        "overlap",
        type=int,
        help="Number of grid cells overlapping between training subdomains.",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help=(
            "First timestep in time series. Provide a string 'YYYYMMDD.HHMMSS'. "
            "If not provided, will use the first timestep in dataset."
        ),
    )
    parser.add_argument(
        "--stop-time",
        type=str,
        default=None,
        help=(
            "Last timestep in time series. Provide a string 'YYYYMMDD.HHMMSS'. "
            "If not provided, will use the last timestep in dataset."
        ),
    )

    return parser


def get_ordered_dims_extent(dims: dict):
    # tile must be first in dims list
    dims_, extent_ = [], []

    for d in ["tile", "x", "y"]:
        dims_.append(d)
        extent_.append(dims.pop(d))
    for k, v in dims.items():
        dims_.append(k)
        extent_.append(v)
    return dims_, extent_


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()

    variables = ["air_temperature", "specific_humidity"]
    data = intake.open_zarr(args.data_path).to_dask()

    tstart = (
        data.time.values[0]
        if not args.start_time
        else vcm.parse_datetime_from_str(args.start_time)
    )
    tstop = (
        data.time.values[-1]
        if not args.stop_time
        else vcm.parse_datetime_from_str(args.stop_time)
    )

    data = data[variables].sel(time=slice(tstart, tstop))
    dims, extent = get_ordered_dims_extent(dict(data.dims))

    cubedsphere_divider = CubedsphereRankDivider(
        tile_layout=(args.layout_width, args.layout_width),
        global_dims=dims,
        global_extent=extent,
    )
    for r in range(cubedsphere_divider.total_ranks):
        rank_output_path = os.path.join(args.output_path, f"rank_{r}.nc")
        rank_data = cubedsphere_divider.get_rank_data(
            data, rank=r, overlap=args.overlap
        )
        with fsspec.open(rank_output_path, "wb") as f:
            rank_data.to_netcdf(f)
