import argparse
import xarray as xr
import numpy as np
import os
import cftime
import pandas as pd


def make_placeholder_data(
    sample: xr.DataArray, generate_variable: str, scaled_factor: float = 1.0
):
    placeholder = xr.DataArray(
        np.random.rand(*sample.values.shape) * scaled_factor,
        coords=sample.coords,
        dims=sample.dims,
    )
    return placeholder.rename(generate_variable)


def make_delp(ps, hyai, hybi):
    p0 = 100000.0
    ps = (
        ps / 2.0
    )  # this round of ps is 2x the actual ps, remove this once we have the updated data
    p_int = p0 * hyai + ps * hybi
    delp = (
        p_int.diff("ilev")
        .rename({"ilev": "lev"})
        .rename("pressure_thickness_of_atmospheric_layer")
    )
    return delp


def make_new_date(dates, year_offset=2015):
    new_dates = []
    for date in dates:
        new_dates.append(
            cftime.DatetimeJulian(
                date.year + year_offset,
                date.month,
                date.day,
                date.hour,
                date.minute,
                date.second,
                date.microsecond,
            )
        )
    return new_dates


def compute_tendencies_due_to_scream_physics(ds: xr.Dataset, nudging_variables: list):
    for var in nudging_variables:
        ds[f"tendency_of_{var}_due_to_scream_physics"] = (
            ds[f"physics_{var}_tend"] - ds[f"nudging_{var}_tend"]
        )
    return ds


def rename_nudging_tendencies(ds: xr.Dataset, nudging_variables: list):
    rename_dict = {}
    for var in nudging_variables:
        rename_dict[f"nudging_{var}_tend"] = f"{var}_tendency_due_to_nudging"
    ds = ds.rename(rename_dict)
    return ds


def rename_delp(ds: xr.Dataset):
    ds = ds.rename({"pseudo_density": "pressure_thickness_of_atmospheric_layer"})
    return ds


def convert_npdatetime_to_cftime(ds: xr.Dataset):
    if isinstance(ds.time.values[0], np.datetime64):
        cf_time = []
        for date in ds.time.values:
            date = pd.to_datetime(date)
            cf_time.append(
                cftime.DatetimeJulian(
                    date.year,
                    date.month,
                    date.day,
                    date.hour,
                    date.minute,
                    date.second,
                )
            )
        ds["time"] = xr.DataArray(cf_time, coords=ds.time.coords, attrs=ds.time.attrs)
    return ds


def rename_lev_to_z(ds: xr.Dataset):
    rename_vars = {"lev": "z", "ilev": "z_interface"}
    rename_vars = {k: v for k, v in rename_vars.items() if k in ds.dims}
    return ds.rename(rename_vars)


def rename_water_vapor_path(ds: xr.Dataset):
    return ds.rename({"VapWaterPath": "water_vapor_path"})


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_data", type=str, default=None, help=("Input netcdf."),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Local or remote path where output will be written.",
    )
    parser.add_argument(
        "nudging_variables",
        type=str,
        help="List of nudging variables deliminated with commas",
    )
    parser.add_argument("chunk_size", type=int, help="Chunk size for output zarrs.")
    parser.add_argument(
        "--split-horiz-winds",
        type=bool,
        default=False,
        help="Split horiz_winds to u and v",
    )
    parser.add_argument(
        "--calc-physics-tend",
        type=bool,
        default=False,
        help="Back out physics only tendencies",
    )
    parser.add_argument(
        "--rename-nudging-tend",
        type=bool,
        default=False,
        help="Rename nudging tendencies",
    )
    parser.add_argument(
        "--rename-delp",
        type=bool,
        default=False,
        help="Rename psuedo density to pressure thickness",
    )

    parser.add_argument(
        "--rename-lev-to-z",
        type=bool,
        default=False,
        help="Rename psuedo density to pressure thickness",
    )

    parser.add_argument(
        "--convert-to-cftime",
        type=bool,
        default=False,
        help="Convert time type to cftime",
    )

    parser.add_argument(
        "--rename-water-vapor-path",
        type=bool,
        default=False,
        help="Rename VapWaterPath to water_vapor_path",
    )
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    ds = xr.open_mfdataset(args.input_data)
    nudging_vars = [str(item) for item in args.nudging_variables.split(",")]
    if args.calc_physics_tend:
        ds = compute_tendencies_due_to_scream_physics(ds, nudging_vars)
    if args.rename_nudging_tend:
        ds = rename_nudging_tendencies(ds, nudging_vars)
    if args.rename_delp:
        ds = rename_delp(ds)
    if args.convert_to_cftime:
        ds = convert_npdatetime_to_cftime(ds)
    if args.rename_lev_to_z:
        ds = rename_lev_to_z(ds)
    if args.rename_water_vapor_path:
        ds = rename_water_vapor_path(ds)
    nudging_variables_tendencies = [
        f"{var}_tendency_due_to_nudging" for var in nudging_vars
    ]
    ds[nudging_variables_tendencies].chunk({"time": args.chunk_size}).to_zarr(
        os.path.join(args.output_path, "nudging_tendencies.zarr"), consolidated=True
    )
    physics_tendencies = [
        f"tendency_of_{var}_due_to_scream_physics" for var in nudging_vars
    ]
    ds[physics_tendencies].chunk({"time": args.chunk_size}).to_zarr(
        os.path.join(args.output_path, "physics_tendencies.zarr"), consolidated=True
    )
    ds_remaining = ds.drop(nudging_variables_tendencies).drop(physics_tendencies)
    ds_remaining.chunk({"time": args.chunk_size}).to_zarr(
        os.path.join(args.output_path, "state_after_timestep.zarr"), consolidated=True
    )
