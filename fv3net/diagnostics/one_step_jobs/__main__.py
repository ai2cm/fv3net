from vcm.cloud.fsspec import get_fs
from fv3net.diagnostics.one_step_jobs.data_funcs_one_step import (
    time_inds_to_open,
    time_coord_to_datetime,
    insert_hi_res_diags,
    insert_derived_vars_from_ds_zarr,
    get_states_and_tendencies,
    insert_column_integrated_tendencies,
    insert_model_run_differences,
    insert_abs_vars, 
    insert_variable_at_model_level,
    insert_weighted_mean_vars,
    shrink_ds
)
from fv3net.pipelines.common import dump_nc
from fv3net import COARSENED_DIAGS_ZARR_NAME
import argparse
import xarray as xr
import os
import logging

ONE_STEP_ZARR = 'big.zarr'
INIT_TIME_DIM = 'initial_time'
FORECAST_TIME_DIM = 'forecast_time'
DELTA_DIM = 'model_run'
GRID_VARS = ['lat', 'lon', 'latb', 'lonb', 'area', 'land_sea_mask']
ABS_VARS = ['psurf', 'precipitable_water', 'total_heat']
GLOBAL_MEAN_2D_VARS = ['psurf_abs', 'precipitable_water_abs', 'total_heat_abs', 'precipitable_water', 'total_heat']
GLOBAL_MEAN_3D_VARS = ["specific_humidity", "air_temperature", "vertical_wind"]
OUTPUT_NC_FILENAME = 'one_step_diag_data.nc'

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "one_step_path", type=str, help="One-step zarr path, including .zarr suffix."
    )
    parser.add_argument(
        "hi_res_diags_path", type=str, help="Output location for diagnostics."
    )
    parser.add_argument(
        "one_step_diags_output_path", type=str, help="Output location for diagnostics."
    )
    parser.add_argument(
        "--start_ind", type=int, default=0, help="First timestep index to use in "
        "zarr. Earlier spin-up timesteps will be skipped. Defaults to 0."
    )
    parser.add_argument(
        "--n_inits_sample", type=int, default=10, help="Number of initalization "
        "to use in computing one-step diagnostics."
    )
    
    return parser


if __name__ == "__main__":
    
    args = _create_arg_parser().parse_args()
    
    zarrpath = os.path.join(args.one_step_path, ONE_STEP_ZARR)
    fs = get_fs(zarrpath)
    ds_zarr = (
        xr
        .open_zarr(fs.get_mapper(zarrpath))
        .isel({INIT_TIME_DIM: slice(args.start_ind, None)})
    )
    logger.info(f"Opened .zarr at {zarrpath}")
    
    timestep_subset_indices = time_inds_to_open(ds_zarr[INIT_TIME_DIM], args.n_inits_sample)
    
    hi_res_diags_zarrpath = os.path.join(args.hi_res_diags_path, COARSENED_DIAGS_ZARR_NAME)
    
    ds_sample = [(
        ds_zarr
        .isel({INIT_TIME_DIM: list(indices)})
        .sel({'step': ['begin', 'after_physics']})
        .pipe(time_coord_to_datetime)
        .pipe(insert_hi_res_diags, hi_res_diags_zarrpath)
        .pipe(insert_derived_vars_from_ds_zarr)
    ) for indices in timestep_subset_indices]
    
    logger.info(f"Sampled {len(ds_sample)} initializations.")
    
#     ds_sample = [
#         ds.assign_coords({INIT_TIME_DIM: time_coord_from_str_coord(ds[INIT_TIME_DIM])})
#         for ds in ds_sample
#     ]
    
    states_and_tendencies = (
        get_states_and_tendencies(ds_sample)
        .pipe(insert_column_integrated_tendencies)
        .pipe(insert_model_run_differences)
        .pipe(insert_abs_vars, ABS_VARS)
        .pipe(insert_variable_at_model_level, ['vertical_wind'], 40)
    )
    
    grid = ds_zarr[GRID_VARS].isel({
        INIT_TIME_DIM: 0,
        FORECAST_TIME_DIM: 0,
        'step': 0
    }).drop_vars(['step', INIT_TIME_DIM, FORECAST_TIME_DIM])
    
    states_and_tendencies = (
        states_and_tendencies
        .merge(grid)
        .pipe(
            insert_weighted_mean_vars,
            grid['area'],
            GLOBAL_MEAN_2D_VARS + GLOBAL_MEAN_3D_VARS
        )
        .pipe(shrink_ds)
    )
    
    print(states_and_tendencies)
    
    output_nc_path = os.path.join(args.one_step_diags_output_path, OUTPUT_NC_FILENAME)
    fs_out = get_fs(output_nc_path)
    
    logger.info(f"Writing stats and tendencies to {output_nc_path}")

    with fs_out.open(output_nc_path, mode="wb") as f:
        dump_nc(states_and_tendencies, f)
