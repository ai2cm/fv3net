import fsspec
import xarray as xr
import os
import intake
from vcm import open_restarts
from vcm.convenience import round_time
import plot
import utils
import argparse
import logging
import sys

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("nudging_diag_animations")

C48_PHYSICS_VARS = [
    "PRATEsfc",
    "SOILM",
    "LHTFLsfc",
    "SHTFLsfc",
    "SOILT1",
    "TMP2m",
    "TMPsfc",
    "SPFH2m",
    "SLMSKsfc",
]
C48_WRAPPER_DIAGS_VARS = ["total_precipitation"]
REFERENCE_RESTART_VARS = ["stc", "smc", "t2m", "q2m"]
REFERENCE_PHYSICS_VARS = [
    "LHTFLsfc_coarse",
    "PRATEsfc_coarse",
    "SHTFLsfc_coarse",
    "tsfc_coarse",
]
GRID_VARS = ["area", "lat", "lon", "latb", "lonb"]
CATALOG = "../../catalog.yml"
RENAME_DIMS = {"grid_xt": "x", "grid_yt": "y"}
GRID_RENAME_DIMS = {
    "grid_xt": "x",
    "grid_yt": "y",
    "grid_x": "x_interface",
    "grid_y": "y_interface",
}
NUDGING_PLOTS = {
    "specific_humidity_tendency_due_to_nudging": {"vmin": -1.5e-7, "vmax": 1.5e-7},
    "air_temperature_tendency_due_to_nudging": {"vmin": -5e-4, "vmax": 5e-4},
}
COMPARISON_PLOTS = {
    "LHTFL": {"abs": {"vmin": -200, "vmax": 200}, "diff": {"vmin": -50, "vmax": 50}},
    "SHTFL": {"abs": {"vmin": -150, "vmax": 150}, "diff": {"vmin": -100, "vmax": 100}},
    "PRATE": {
        "abs": {"vmin": 0.0, "vmax": 20.0},
        "diff": {"vmin": -15.0, "vmax": 15.0},
    },
    "SOILM": {
        "abs": {"vmin": 0.0, "vmax": 1000.0},
        "diff": {"vmin": -50.0, "vmax": 50.0},
    },
    "TMP2m": {
        "abs": {"vmin": 240.0, "vmax": 300.0},
        "diff": {"vmin": -3.0, "vmax": 3.0},
    },
    "SPFH2m": {
        "abs": {"vmin": 0.0, "vmax": 0.02},
        "diff": {"vmin": -0.001, "vmax": 0.001},
    },
}
INTERVAL = 100


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nudging_root_path",
        type=str,
        help="Path to directory containing output zarrs from nudging run.",
    )
    parser.add_argument(
        "reference_restart_path",
        type=str,
        help="Path to directory containing reference run restart files.",
    )
    parser.add_argument(
        "reference_gfs_physics_zarr_path",
        type=str,
        help=(
            "Either path to directory containing reference gfs physics zarr, "
            "or name of catalog entry containing such."
        ),
    )
    parser.add_argument(
        "output_path", type=str, help="Path to which output animations will be copied."
    )
    parser.add_argument(
        "-timestep_stride",
        type=int,
        default=3,
        help="Stride in C48 run data for makinganimations and analysis.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Boolean flag to handle a baseline run, i.e., don't replace PRATE "
            "with total_preciptiation"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":

    logger.info("Starting routine.")

    args = _create_arg_parser()

    catalog = intake.open_catalog(CATALOG)

    (surface_zarr_name, nudging_plots) = (
        ("sfc_dt_atmos.zarr", False) if args.baseline else ("physics_output.zarr", True)
    )

    # first, figure out what timesteps are available in the nudging run
    nudging_physics_path = os.path.join(args.nudging_root_path, surface_zarr_name)
    nudged_C48_physics_ds = utils.drop_uninformative_coords(
        utils.remove_suffixes(
            xr.open_zarr(fsspec.get_mapper(nudging_physics_path))[
                C48_PHYSICS_VARS + GRID_VARS
            ]
        )
    )

    nudged_C48_physics_ds = nudged_C48_physics_ds.assign_coords(
        {"time": [round_time(time.item()) for time in nudged_C48_physics_ds.time]}
    )

    # decide what frequency to sample -- most frequent is 2 hrs
    nudging_subset_times = nudged_C48_physics_ds.time.isel(
        time=slice(None, None, args.timestep_stride)
    )
    timestamps = [
        time.item().strftime("%Y%m%d.%H%M%S") for time in nudging_subset_times
    ]

    # extract grid and data variables as separate datasets
    grid_ds = nudged_C48_physics_ds.sel(time=nudging_subset_times)[
        GRID_VARS + ["SLMSK"]
    ].rename(GRID_RENAME_DIMS)
    nudged_C48_physics_ds = nudged_C48_physics_ds.sel(
        time=nudging_subset_times
    ).drop_vars(names=GRID_VARS + ["SLMSK"])

    if not args.baseline:

        # open nudging tendencies as well
        nudging_tendencies_path = os.path.join(
            args.nudging_root_path, "nudging_tendencies.zarr"
        )
        nudged_C48_tendencies_ds = (
            xr.open_zarr(fsspec.get_mapper(nudging_tendencies_path))
        ).sel(time=nudging_subset_times)

        # open wrapper diagnostics to get 'total_precipitation'
        # and insert it as 'PRATE'
        nudging_states_before_dynamics_path = os.path.join(
            args.nudging_root_path, "after_dynamics.zarr"
        )
        nudging_C48_after_dynamics_ds = (
            (
                xr.open_zarr(fsspec.get_mapper(nudging_states_before_dynamics_path))[
                    C48_WRAPPER_DIAGS_VARS
                ]
            )
            .sel(time=nudging_subset_times)
            .rename({"total_precipitation": "PRATE"})
        )
        nudged_C48_physics_ds = nudged_C48_physics_ds.drop_vars(names=["PRATE"])

        # merge nudging data
        nudging_ds_list = [
            nudged_C48_physics_ds.rename(RENAME_DIMS),
            nudged_C48_tendencies_ds,
            nudging_C48_after_dynamics_ds,
        ]
        nudged_C48_ds = xr.merge(nudging_ds_list)
    else:
        nudged_C48_ds = nudged_C48_physics_ds.rename(RENAME_DIMS)

    # then, open coarsened reference restart files matching these times
    coarsened_reference_restart_ds = (
        utils.drop_uninformative_coords(
            xr.concat(
                [
                    open_restarts(os.path.join(args.reference_restart_path, timestamp))
                    for timestamp in timestamps
                ],
                dim="time",
            ).assign_coords({"time": nudging_subset_times})
        )[REFERENCE_RESTART_VARS]
        .rename(RENAME_DIMS)
        .squeeze()
    )

    coarsened_reference_restart_ds = (
        coarsened_reference_restart_ds.pipe(utils.sum_soil_moisture)
        .pipe(utils.mask_soilm_to_land, grid_ds)
        .pipe(utils.top_soil_temperature_only)
        .drop_vars(names=["file_prefix", "soil_layer"])
    )

    # open coarsened gfs_physics zarr from reference run
    reference_gfs_physics_zarr_path = args.reference_gfs_physics_zarr_path
    if reference_gfs_physics_zarr_path.startswith("gs://"):
        coarsened_reference_physics_ds = xr.open_zarr(
            fsspec.get_mapper(reference_gfs_physics_zarr_path)
        )
    elif reference_gfs_physics_zarr_path.startswith("40day"):
        coarsened_reference_physics_ds = catalog[
            reference_gfs_physics_zarr_path
        ].to_dask()

    coarsened_reference_physics_ds = coarsened_reference_physics_ds.assign_coords(
        {
            "time": [
                round_time(time.item()) for time in coarsened_reference_physics_ds.time
            ]
        }
    )

    coarsened_reference_physics_ds = utils.drop_uninformative_coords(
        utils.remove_suffixes(
            coarsened_reference_physics_ds.sel(time=nudging_subset_times)[
                REFERENCE_PHYSICS_VARS
            ]
        )
    ).rename(RENAME_DIMS)

    # merge and rename the reference datasets
    coarsened_reference_ds = utils.rename_reference_vars(
        xr.merge([coarsened_reference_physics_ds, coarsened_reference_restart_ds])
    )

    # put the nudged and reference datasets together and compute differences
    ds = (
        utils.concat_and_differences(coarsened_reference_ds, nudged_C48_ds)
        .assign_coords({"derivation": ["coarsened_reference", "C48", "difference"]})
        .merge(grid_ds)
        .pipe(utils.precip_units)
    ).load()

    # add land averages
    for var in COMPARISON_PLOTS:
        ds = ds.assign(
            {
                f"{var}_land_average": utils.global_average(
                    ds[var], ds["area"], ds["SLMSK"], "land"
                )
            }
        )

    ds.to_netcdf(os.path.join(args.output_path, "biases.nc"))

    logger.info("Finished preprocessing routine.")

    # make animations
    if not args.baseline:
        for variable, scale in NUDGING_PLOTS.items():
            logger.info(f"Making animation for {variable}.")
            anim = plot.make_nudging_animation(
                ds, variable, interval=INTERVAL, plot_cube_kwargs=scale
            )
            anim.save(os.path.join(args.output_path, f"{variable}.mp4"))
    for variable, scales in COMPARISON_PLOTS.items():
        logger.info(f"Making animation for {variable}.")
        anim = plot.make_comparison_animation(
            ds,
            variable,
            interval=INTERVAL,
            plot_cube_kwargs_abs=scales["abs"],
            plot_cube_kwargs_diff=scales["diff"],
        )
        anim.save(os.path.join(args.output_path, f"{variable}.mp4"))
