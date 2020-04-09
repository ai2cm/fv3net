from vcm.cloud.fsspec import get_fs, get_protocol
from vcm.cloud.gsutil import copy
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
    insert_diurnal_means,
    insert_area_means,
    shrink_ds
)
from fv3net.diagnostics.one_step_jobs.plotting_funcs_one_step import make_all_plots
from fv3net.diagnostics.one_step_jobs import (
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
    STEP_DIM,
    ONE_STEP_ZARR,
    OUTPUT_NC_FILENAME,
    SFC_VARIABLES,
    GRID_VARS,
    VARS_FROM_ZARR,
    ABS_VARS,
    GLOBAL_MEAN_2D_VARS,
    GLOBAL_MEAN_3D_VARS,
    DIURNAL_VAR_MAPPING
)
from fv3net.diagnostics.create_report import create_report
from fv3net.pipelines.common import dump_nc
from fv3net import COARSENED_DIAGS_ZARR_NAME
import argparse
import xarray as xr
import numpy as np
import os
import logging
import sys


logger = logging.getLogger("one_step_diags")
logger.setLevel(logging.INFO)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter(
    '%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s'
))
out_hdlr.setLevel(logging.INFO)
logger.addHandler(out_hdlr)


def _create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "one_step_data", type=str, help="One-step zarr path, including .zarr suffix."
    )
    parser.add_argument(
        "hi_res_diags", type=str, help="Output location for diagnostics."
    )
    parser.add_argument(
        "netcdf_output", type=str, help="Output location for diagnostics netcdf file."
    )
    parser.add_argument(
        "--report_directory", type=str, default=None,
        help="(Public) bucket path for report and image upload. If omitted, report is"
        "written to netcdf_output."
    )
    parser.add_argument(
        "--start_ind", type=int, default=0, help="First timestep index to use in "
        "zarr. Earlier spin-up timesteps will be skipped. Defaults to 0."
    )
    parser.add_argument(
        "--n_sample_inits", type=int, default=10, help="Number of initalization "
        "to use in computing one-step diagnostics."
    )
    
    return parser


if __name__ == "__main__":
    
    args = _create_arg_parser().parse_args()
    
    zarrpath = os.path.join(args.one_step_data, ONE_STEP_ZARR)
    fs = get_fs(zarrpath)
    ds_zarr = (
        xr
        .open_zarr(fs.get_mapper(zarrpath))
        .isel({INIT_TIME_DIM: slice(args.start_ind, None)})
        [list(VARS_FROM_ZARR + GRID_VARS)]
    )
    logger.info(f"Opened .zarr at {zarrpath}")
    
    timestep_subset_indices = time_inds_to_open(ds_zarr[INIT_TIME_DIM], args.n_sample_inits)
    
    hi_res_diags_zarrpath = os.path.join(args.hi_res_diags, COARSENED_DIAGS_ZARR_NAME)
    hi_res_diags_mapping = {name: name for name in SFC_VARIABLES}
    hi_res_diags_mapping.update({
        'latent_heat_flux': 'LHTFLsfc',
        'sensible_heat_flux': 'SHTFLsfc',
        'total_precipitation': 'PRATEsfc'
    })
    
    ds_sample = [(
        ds_zarr[list(VARS_FROM_ZARR)]
        .isel({INIT_TIME_DIM: list(indices)})
        .sel({STEP_DIM: ['begin', 'after_physics']})
        .assign_coords({'z': np.arange(1., ds_zarr.sizes['z'] + 1.)})
        .pipe(time_coord_to_datetime)
        .pipe(insert_hi_res_diags, hi_res_diags_zarrpath, hi_res_diags_mapping)
        .pipe(insert_derived_vars_from_ds_zarr)
    ) for indices in timestep_subset_indices]
    
    logger.info(f"Sampling {len(ds_sample)} initializations at: "
                f"{[ds[INIT_TIME_DIM].values[0] for ds in ds_sample]}")
    
    states_and_tendencies = (
        get_states_and_tendencies(ds_sample)
        .pipe(insert_column_integrated_tendencies)
        .pipe(insert_model_run_differences)
        .pipe(insert_abs_vars, ABS_VARS)
        .pipe(insert_variable_at_model_level, ['vertical_wind'], 40)
    )
    
    grid = ds_zarr[list(GRID_VARS)].isel({
        INIT_TIME_DIM: 0,
        FORECAST_TIME_DIM: 0,
        STEP_DIM: 0
    }).drop_vars([STEP_DIM, INIT_TIME_DIM, FORECAST_TIME_DIM])
    
    logger.info('Subsetting dataset and loading')
    
    states_and_tendencies = (
        states_and_tendencies
        .merge(grid)
        .pipe(insert_diurnal_means)
        .pipe(
            insert_area_means,
            grid['area'],
            GLOBAL_MEAN_2D_VARS + GLOBAL_MEAN_3D_VARS,
            ['land_sea_mask', 'net_precipitation_physics']
        )
        .pipe(shrink_ds)
        .load()
    )
    
    output_path = args.netcdf_output
    output_nc_path = os.path.join(output_path, OUTPUT_NC_FILENAME)
    fs_out = get_fs(output_nc_path)
    
    logger.info(f"Writing states and tendencies to {output_nc_path}")

    with fs_out.open(output_nc_path, mode="wb") as f:
        dump_nc(states_and_tendencies, f)
        
    logger.info(f"Writing diagnostics plots report to {output_nc_path}")
    
    
    # if output path is remote GCS location, save results to local output dir first
    if args.report_directory:
        report_path = args.report_directory
    else:
        report_path = output_path
    proto = get_protocol(report_path)
    if proto == "" or proto == "file":
        output_dir = report_path
    elif proto == "gs":
        remote_data_path, output_dir = os.path.split(report_path.strip("/"))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    report_sections = make_all_plots(states_and_tendencies, output_dir)
    create_report(report_sections, "one_step_diagnostics", output_dir)
    if proto == "gs":
        copy(output_dir, remote_data_path)
