import argparse
import xarray as xr
import numpy as np
import os
import cftime
import util


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


def rename_water_vapor_path(ds: xr.Dataset):
    return ds.rename({"VapWaterPath": "water_vapor_path"})


def add_rad_fluxes(ds: xr.Dataset):
    shortwave_transmissivity_of_atmospheric_column = (
        ds.SW_flux_dn_at_model_bot / ds.SW_flux_dn_at_model_top
    )
    shortwave_transmissivity_of_atmospheric_column = shortwave_transmissivity_of_atmospheric_column.where(  # noqa
        ds.SW_flux_dn_at_model_top != 0.0, 0.0
    )
    shortwave_transmissivity_of_atmospheric_column = shortwave_transmissivity_of_atmospheric_column.assign_attrs(  # noqa
        units="-", long_name="shortwave transmissivity of atmosphericcolumn"
    )
    override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface = (  # noqa
        ds.LW_flux_dn_at_model_bot
    )
    override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface = override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface.assign_attrs(  # noqa
        units="W/m**2", long_name="surface downward longwave flux"
    )
    ds[
        "shortwave_transmissivity_of_atmospheric_column"
    ] = shortwave_transmissivity_of_atmospheric_column
    ds[
        "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface"
    ] = override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface
    return ds


def add_sfc_geopotential_height(
    ds: xr.Dataset,
    path="/usr/gdata/climdat/ccsm3data/inputdata/atm/cam/topo/USGS-gtopo30_ne30np4pg2_x6t-SGH.c20210614.nc",  # noqa
):
    sfc_geo = xr.open_dataset(path)
    phis = sfc_geo.PHIS
    phis = phis.expand_dims(dim={"time": ds.time}, axis=0)
    ds["surface_geopotential"] = phis
    return ds


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
    parser.add_argument(
        "tend_processes",
        type=str,
        help="List of tendency processes deliminated with commas",
    )
    parser.add_argument("chunk_size", type=int, help="Chunk size for output zarrs.")
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

    parser.add_argument(
        "--split-horiz-winds-tend",
        type=bool,
        default=False,
        help="Rename horiz_winds ",
    )
    parser.add_argument(
        "--add-rad-fluxes",
        type=bool,
        default=False,
        help="Add derived radiative fluxes",
    )
    parser.add_argument(
        "--add-phis", type=bool, default=False, help="Add surface geopotential height",
    )
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    ds = xr.open_mfdataset(args.input_data)
    nudging_vars = [str(item) for item in args.nudging_variables.split(",")]
    if args.split_horiz_winds_tend:
        tend_processes = [str(item) for item in args.tend_processes.split(",")]
        for i in range(len(tend_processes)):
            print(f"Splitting {tend_processes[i]} horiz winds tendency")
            ds = util.split_horiz_winds_tend(ds, tend_processes[i])
    if args.calc_physics_tend:
        print("Calculating scream physics tendencies")
        ds = compute_tendencies_due_to_scream_physics(ds, nudging_vars)
    if args.rename_nudging_tend:
        print("Renaming nudging tendencies")
        ds = rename_nudging_tendencies(ds, nudging_vars)
    if args.rename_delp:
        print("Renaming nudging delp")
        ds = rename_delp(ds)
    if args.convert_to_cftime:
        print("Converting timestamps to cftime")
        ds = util.convert_npdatetime_to_cftime(ds)
    if args.rename_lev_to_z:
        print("Renaming lev to z")
        ds = util.rename_lev_to_z(ds)
    if args.rename_water_vapor_path:
        print("Renaming water vapor path")
        ds = rename_water_vapor_path(ds)
    if args.add_rad_fluxes:
        print("Adding derived radiative fluxes")
        ds = add_rad_fluxes(ds)
    if args.add_phis:
        print("Adding surface geopotential height")
        ds = add_sfc_geopotential_height(ds)
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
