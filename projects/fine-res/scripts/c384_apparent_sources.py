import logging
from datetime import timedelta

import click
import fsspec
import numpy as np
import xarray as xr
from dask.distributed import Client
from vcm.calc import convergence_cell_center
from vcm.convenience import parse_datetime_from_str, round_time
from vcm.fv3.metadata import gfdl_to_standard
from vcm.safe import get_variables

PHYSICS_VARIABLES = [
    "t_dt_fv_sat_adj_coarse",
    "t_dt_nudge_coarse",
    "t_dt_phys_coarse",
    "qv_dt_fv_sat_adj_coarse",
    "qv_dt_phys_coarse",
    "eddy_flux_vulcan_omega_sphum",
    "eddy_flux_vulcan_omega_temp",
    "vulcan_omega_coarse",
]
DELP = "delp"
C384_ATMOS = "gs://vcm-ml-raw/2020-05-27-40-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr"  # noqa: E501
C384_RESTARTS = "gs://vcm-ml-experiments/2020-06-02-fine-res/2020-05-27-40-day-X-SHiELD-simulation-C384-restart-files.zarr"  # noqa: E501
OUTPUT_URL = "gs://vcm-ml-intermediate/2022-01-04-c384-fine-res-budget-from-2020-05-27-40-day-X-SHiELD-simulation.zarr"  # noqa: E501


def open_diags(path: str):
    ds = xr.open_zarr(fsspec.get_mapper(path), consolidated=True)
    times = round_time(ds.time)
    return (
        ds.assign_coords(time=times)
        .pipe(rename_dims)
        .pipe(rename_latlon)
        .drop_vars("pfull")
    )


def rename_dims(ds):
    name_dict = {}
    for variable in ds.dims:
        suffix = "_coarse"
        if variable.endswith(suffix):
            name_dict[variable] = variable[: -len(suffix)]
    return ds.rename(name_dict)


def rename_latlon(ds):
    return ds.rename(
        {
            "grid_lat_coarse": "latb",
            "grid_lon_coarse": "lonb",
            "grid_lont_coarse": "lon",
            "grid_latt_coarse": "lat",
        }
    )


def open_restarts(path: str):
    ds = xr.open_zarr(fsspec.get_mapper(path), consolidated=True)
    times = np.vectorize(parse_datetime_from_str)(ds.time)
    return ds.assign(time=times).drop_vars("tile", errors="ignore")


def shift_to_time_average(ds, dt=timedelta(seconds=30, minutes=7)):
    time = ds.time
    begin = ds.assign_coords(time=time + dt)
    end = ds.assign_coords(time=time - dt)
    return (begin + end) / 2


def apparent_moistening(data, vdim="pfull"):
    eddy_flux_convergence = convergence_cell_center(
        data.eddy_flux_vulcan_omega_sphum, data.delp, dim=vdim
    )
    return (
        (data.qv_dt_fv_sat_adj_coarse + data.qv_dt_phys_coarse + eddy_flux_convergence)
        .assign_attrs(
            units="kg/kg/s",
            long_name="apparent moistening from fine-grid data",
            description=(
                "Apparent moistening due to physics and sub-grid-scale advection. "
                "Given by "
                "sat adjustment (dycore) + physics tendency + eddy-flux-convergence"
            ),
        )
        .rename("Q2")
    )


def apparent_heating(data, include_temperature_nudging: bool = False, vdim="pfull"):
    eddy_flux_convergence = convergence_cell_center(
        data.eddy_flux_vulcan_omega_temp, data.delp, dim=vdim
    )
    result = data.t_dt_fv_sat_adj_coarse + data.t_dt_phys_coarse + eddy_flux_convergence
    description = (
        "Apparent heating due to physics and sub-grid-scale advection. Given "
        "by sat adjustment (dycore) + physics tendency + eddy-flux-convergence"
    )
    if include_temperature_nudging:
        result = result + data.t_dt_nudge_coarse
        description = description + " + temperature nudging"
    return result.assign_attrs(
        units="K/s",
        long_name="apparent heating from fine-grid data",
        description=description,
    ).rename("Q1")


def standardize_coords(ds: xr.Dataset, time_shift=timedelta(minutes=7, seconds=30)):
    ds_shifted = ds.assign(time=ds.time + time_shift)
    return gfdl_to_standard(ds_shifted).drop("tile")


@click.command()
@click.argument("output_url", type=str, default=OUTPUT_URL)
@click.argument("c384_atmos_dt_atmos", type=str, default=C384_ATMOS)
@click.argument("c384_restarts", type=str, default=C384_RESTARTS)
@click.option(
    "--i_start_time",
    type=int,
    default=0,
    help=(
        "0-based integer index of first 15 minute timestep to include,"
        " starting at 20160801.0015000."
    ),
)
@click.option(
    "--i_end_time",
    type=int,
    default=None,
    help=(
        "0-based integer index of first 15 minute timestep after end of desired time,"
        " starting at 20160801.0015000."
    ),
)
@click.option("--include_temperature_nudging", type=bool, default=True)
@click.option(
    "--append_dim",
    type=str,
    default=None,
    help="dimension (e.g., 'time'), along which to append to an existing zarr.",
)
def main(
    output_url,
    c384_atmos_dt_atmos,
    c384_restarts,
    i_start_time,
    i_end_time,
    include_temperature_nudging,
    append_dim,
):

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting C384 apparent sources computation.")

    client = Client()
    logging.info(f"See dask client dashboard: {client.dashboard_link}")

    diags = open_diags(c384_atmos_dt_atmos)
    restarts = open_restarts(c384_restarts)
    delp = shift_to_time_average(restarts[DELP])

    merged = xr.merge([get_variables(diags, PHYSICS_VARIABLES), delp], join="inner")
    Q1, Q2 = (
        apparent_heating(
            merged, include_temperature_nudging=include_temperature_nudging
        ),
        apparent_moistening(merged),
    )
    output_ds = standardize_coords(xr.Dataset({"Q1": Q1, "Q2": Q2}))
    output_ds = output_ds.isel(time=slice(i_start_time, i_end_time))
    history = (
        f"Saved to {output_url} using fv3net/projects"
        "/fine-res/scripts/c384_apparent_sources.py"
    )
    output_ds.attrs["history"] = history

    logging.info(f"Saving to {output_url}.")
    output_ds.to_zarr(
        fsspec.get_mapper(output_url), consolidated=True, append_dim=append_dim
    )


if __name__ == "__main__":
    main()
