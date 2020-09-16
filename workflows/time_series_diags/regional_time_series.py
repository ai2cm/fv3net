import argparse
from datetime import datetime
import fsspec
import intake
import logging
import os
import pandas as pd
import tempfile
from typing import Sequence
import xarray as xr
import yaml
import zarr.storage as zstore

import _utils as utils
import loaders
from report import create_html
from vcm import RegionOfInterest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vertical_profile_plots")
xr.set_options(keep_attrs=True)

TIME_FMT = "%Y%m%d.%H%M%S"
RENAME_VARS = {
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
DATA_VARS = [
    "pressure_thickness_of_atmospheric_layer",
    "air_temperature",
    "specific_humidity",
    "pQ1",
    "pQ2",
]


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_data_path", nargs="*", type=str, help="Location of run data."
    )
    parser.add_argument(
        "fine_res_reference_path", type=str, help="Location of reference fine res data."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help=("Local or remote path where diagnostic dataset will be written."),
    )
    parser.add_argument(
        "--lat-bounds", nargs=2, type=float, help=("Min, max latitude bounds"),
    )
    parser.add_argument(
        "--lon-bounds", nargs=2, type=float, help=("Min, max longitude bounds"),
    )
    parser.add_argument(
        "--consolidated",
        type=bool,
        default=False,
        help="Is zarr metadata consolidated?",
    )
    parser.add_argument(
        "--time-bounds",
        nargs=2,
        type=str,
        help="Optional, min/max time range. Should have format 'YYYYMMDD.HHMMSS'.",
    )
    parser.add_argument(
        "--train-data-config",
        type=str,
        help="Optional, provide if reading vertical profiles from training data.",
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default="catalog.yml",
        help="Path to catalog from where script is executed",
    )
    return parser.parse_args()


def _dataset_from_zarr(url: str, time_bounds: Sequence[str] = None, consolidated: bool = False):
    mapper = fsspec.get_mapper(url)
    if time_bounds:
        time_slice = slice(*[datetime.strptime(t, TIME_FMT) for t in time_bounds])
    else:
        time_slice = slice(None, None)
    ds = xr.open_zarr(
        zstore.LRUStoreCache(mapper, 1024),
        consolidated=consolidated,
        mask_and_scale=False,
    )
    renamed = {key: value for key, value in RENAME_VARS.items() if key in ds.data_vars}
    ds = (
        ds.rename(renamed)[DATA_VARS]
        .pipe(utils.standardize_zarr_time_coord)
        .sel({"time": time_slice})
    )
    return ds


def _fine_res_reference(fine_res_path: str, times: Sequence[datetime]):
    mapper = loaders.mappers.open_fine_res_apparent_sources(
        fine_res_path, offset_seconds=450
    )
    times = [pd.to_datetime(t).strftime(TIME_FMT) for t in times]
    return utils.dataset_from_timesteps(
        mapper, times, ["air_temperature", "specific_humidity"]
    )


def _dataset_from_training_config(data_paths, config, time_bounds):
    mapper_func = getattr(loaders.mappers, config["batch_kwargs"]["mapping_function"])
    data_path = args.run_data_path
    if len(data_path) == 1:
        data_path = data_path[0]
    mapper = mapper_func(data_path, **config["batch_kwargs"].get("mapping_kwargs", {}))
    times = (
        list(mapper.keys())
        if not time_bounds
        else utils.time_range_str_format(list(mapper.keys()), time_bounds)
    )
    return utils.dataset_from_timesteps(mapper, times, DATA_VARS).sortby("time")


def _open_dataset(args):
    if ".zarr" in args.run_data_path:
        ds = _dataset_from_zarr(args.run_data_path, args.time_bounds, args.consolidated,)
    elif args.train_data_config:
        with fsspec.open(args.train_data_config, "r") as f:
            config = yaml.safe_load(f)
        ds = _dataset_from_training_config(args.run_data_path, config, args.time_bounds)
    else:
        raise ValueError(
            "Provide either i) a zarr as the arg run_data_path or "
            "ii) a training configuration file that has mapper information "
            "for training data as --train-data-config.")
    return ds


if __name__ == "__main__":
    args = _create_arg_parser()

    cat = intake.open_catalog(args.catalog_path)
    grid = cat["grid/c48"].to_dask()
    ds = _open_dataset(args).merge(grid)
    fine_res = _fine_res_reference(args.fine_res_reference_path, ds.time.values)

    for var in ["air_temperature", "specific_humidity"]:
        ds[f"{var}_anomaly"] = ds[var] - fine_res[var]

    if args.lat_bounds and args.lon_bounds:
        region = RegionOfInterest(tuple(args.lat_bounds), tuple(args.lon_bounds))
    else:
        # default to equatorial zone
        region = RegionOfInterest(lat_bounds=[-10, 10], lon_bounds=[0, 360])

    ds = region.average(ds)
    ds = utils.insert_pressure_level_temp(ds)

    metadata = {
        "run data": args.run_data_path,
        "fine res data": args.fine_res_reference_path,
        "lat bounds": region.lat_bounds,
        "lon bounds": region.lon_bounds,
        "time min/max": args.time_bounds,
    }
    figure_paths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for var in [
            "air_temperature_anomaly",
            "specific_humidity_anomaly",
            "pQ1",
            "pQ2",
            "T850",
            "T200",
            "T850-T200",
        ]:
            fig = utils.time_series(ds[var], grid)
            fig_name = f"{var}_time_series.png"
            figure_paths.append(fig_name)
            outfile = os.path.join(tmpdir, fig_name)
            fig.savefig(outfile)
            logger.info(f"Saved figure {var} time_series.")
        sections = {"Time series": figure_paths}
        html = create_html(sections, "", metadata)
        with open(os.path.join(tmpdir, "time_series.html"), "w") as f:
            f.write(html)
        utils.copy_outputs(tmpdir, args.output_dir)
