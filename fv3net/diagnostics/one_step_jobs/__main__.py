from vcm.cloud.fsspec import get_fs, get_protocol
from vcm.cloud.gsutil import copy
from vcm.cubedsphere.constants import TIME_FMT
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
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import argparse
import xarray as xr
import numpy as np
import os
import shutil
from tempfile import TemporaryDirectory
import logging
import sys
from typing import Sequence, Mapping, Any

out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter(
    '%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s'
))
out_hdlr.setLevel(logging.INFO)
logging.basicConfig(handlers=[out_hdlr])
logger = logging.getLogger("one_step_diags")


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


def _insert_derived_vars(
    ds: xr.Dataset,
    hi_res_diags_zarrpath: Sequence,
    hi_res_diags_mapping: Mapping
) -> xr.Dataset:
    """dataflow pipeline func for adding derived variables to the raw dataset
    """
    
    try: 
        logger.info(f"Inserting derived variables for timestep "
                     f"{ds[INIT_TIME_DIM].values[0]}")
        ds = (
            ds
            .assign_coords({'z': np.arange(1., ds_zarr.sizes['z'] + 1.)})
            .pipe(time_coord_to_datetime)
            .pipe(insert_hi_res_diags, hi_res_diags_zarrpath, hi_res_diags_mapping)
            .pipe(insert_derived_vars_from_ds_zarr)
        )
    except Exception as e:
        logger.warning(e)
        ds = None
        
    return ds


def _insert_states_and_tendencies(ds: xr.Dataset) -> xr.Dataset:
    """dataflow pipeline func for adding states and tendencies
    """

    try:
        logger.info(f"Inserting states and tendencies for timestep "
                     f"{ds[INIT_TIME_DIM].values[0].strftime(TIME_FMT)}")
        ds = (
            get_states_and_tendencies(ds)
            .pipe(insert_column_integrated_tendencies)
            .pipe(insert_model_run_differences)
            .pipe(insert_abs_vars, ABS_VARS)
            .pipe(insert_variable_at_model_level, ['vertical_wind'], 40)
        )
        ds.attrs[INIT_TIME_DIM] = [ds[INIT_TIME_DIM].item().strftime(TIME_FMT)]
    except Exception as e:
        ds = None
        logger.warning(e)
    
    return ds


def _insert_means_and_shrink(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
    """dataflow pipeline func for adding mean variables and shrinking dataset to
    managable size for aggregation and saving
    """
    
    try:
        timestep = ds[INIT_TIME_DIM].item().strftime(TIME_FMT)
        logger.info(f"Inserting means and shrinking timestep {timestep}")
        ds = (
            ds
            .merge(grid)
            .pipe(insert_diurnal_means)
            .pipe(
                insert_area_means,
                grid['area'],
                GLOBAL_MEAN_2D_VARS.keys() + GLOBAL_MEAN_3D_VARS,
                ['land_sea_mask', 'net_precipitation_physics']
            )
            .pipe(shrink_ds)
        )
        logger.info(f"Finished shrinking timestep {timestep}")
    except Exception as e:
        ds = None
        logger.warning(e)
        
    return ds


def _filter_ds(ds: Any):
    return isinstance(ds, xr.Dataset)


def _merge_ds(pc: Sequence[xr.Dataset]) -> xr.Dataset:
    """dataflow pipeline func for aggregating datasets across initialization time
    """
    
    try:
        logger.info("Aggregating data across timesteps.")
        inits = []
        for ds in pc:
            if isinstance(ds, xr.Dataset):
                inits = list(set(inits).union(set(ds.attrs[INIT_TIME_DIM])))
        ds = xr.concat(pc, dim = INIT_TIME_DIM)
        ds.attrs[INIT_TIME_DIM] = inits
    except Exception as e:
        logger.warning(e)
        ds = []

    return ds


def _mean_and_std(ds: xr.Dataset) -> xr.Dataset:
    """dataflow pipeline func for taking mean and standard deviation along
    initialization time dimension of aggregated diagnostic dataset
    """
    try:
        logger.info("Computing aggregated means and std. devs.")
        for var in ds:
            if INIT_TIME_DIM in ds[var].dims and var not in GRID_VARS:
                var_std = ds[var].std(dim = INIT_TIME_DIM, keep_attrs=True)
                if 'long_name' in var_std.attrs:
                    var_std = var_std.assign_attrs(
                        {'long_name': f"{var_std.attrs['long_name']} std. dev."}
                    )
                ds = ds.assign({f"{var}_std": var_std})
        ds_mean = ds.mean(dim=INIT_TIME_DIM, keep_attrs=True)
    except Exception as e:
        logger.info(e)
        ds_mean = None
        
    return ds_mean


def _write_ds(ds: xr.Dataset, fullpath: str):
    """dataflow pipeline func for writing out final netcdf"""
    
    logger.info("fWriting final dataset to netcdf at {fullpath}.")
    for var in ds.variables:
        if ds[var].dtype == 'O':
            ds = ds.assign({var: ds[var].astype('S15').astype('unicode_')})
    with TemporaryDirectory() as tmpdir:
        pathname, filename = os.path.split(fullpath)
        tmppath = os.path.join(tmpdir, filename)
        ds.to_netcdf(tmppath)
        copy(tmppath, fullpath)


if __name__ == "__main__":
    
    args, pipeline_args = _create_arg_parser().parse_known_args()
    
    zarrpath = os.path.join(args.one_step_data, ONE_STEP_ZARR)
    fs = get_fs(zarrpath)
    ds_zarr = (
        xr
        .open_zarr(fs.get_mapper(zarrpath))
        .isel({INIT_TIME_DIM: slice(args.start_ind, None)})
        [list(VARS_FROM_ZARR + GRID_VARS)]
    )
    logger.info(f"Opened .zarr at {zarrpath}")
    
    timestamp_subset_indices = time_inds_to_open(ds_zarr[INIT_TIME_DIM], args.n_sample_inits)
    
    ds_sample = [(
        ds_zarr[list(VARS_FROM_ZARR)]
        .isel({INIT_TIME_DIM: list(indices)})
        .sel({'step': list(('begin', 'after_physics'))})
    ) for indices in timestamp_subset_indices]
    
    hi_res_diags_zarrpath = os.path.join(args.hi_res_diags, COARSENED_DIAGS_ZARR_NAME)
    hi_res_diags_mapping = {name: name for name in SFC_VARIABLES}
    hi_res_diags_mapping.update({
        'latent_heat_flux': 'LHTFLsfc',
        'sensible_heat_flux': 'SHTFLsfc',
        'total_precipitation': 'PRATEsfc'
    })
    
    grid = ds_zarr[list(GRID_VARS)].isel({
        INIT_TIME_DIM: 0,
        FORECAST_TIME_DIM: 0,
        STEP_DIM: 0
    }).drop_vars([STEP_DIM, INIT_TIME_DIM, FORECAST_TIME_DIM])

    
#     output_path = args.netcdf_output
#     if proto == "" or proto == "file":
#         output_nc_dir = output_path
#     elif proto == "gs":
#         remote_data_path, output_nc_dir = os.path.split(output_path.strip("/"))
#     if os.path.exists(output_nc_dir):
#         shutil.rmtree(output_nc_dir)
#     os.mkdir(output_nc_dir)
    
    output_nc_path = os.path.join(args.netcdf_output, OUTPUT_NC_FILENAME)

    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | "CreateDS" >> beam.Create(ds_sample)
            | "InsertDerivedVars" >> beam.Map(_insert_derived_vars, hi_res_diags_zarrpath, hi_res_diags_mapping)
            | "InsertStatesAndTendencies" >> beam.Map(_insert_states_and_tendencies)
            | "InsertMeanVars" >> beam.Map(_insert_means_and_shrink, grid)
            | "FilterDS" >> beam.Filter(_filter_ds) 
            | "MergeDS" >> beam.CombineGlobally(_merge_ds) 
            | "MeanAndStdDev" >> beam.Map(_mean_and_std)
            | "WriteDS" >> beam.Map(_write_ds, output_nc_path)
        )
        
    proto = get_protocol(output_nc_path)
    if proto == "gs":
        fs_nc = get_fs(output_nc_path)
        states_and_tendencies = xr.open_dataset(fs_nc.get_mapper(output_nc_path))
    elif proto == "" or proto == "file":
        states_and_tendencies = xr.open_dataset(output_nc_path)
    
    # if report output path is GCS location, save results to local output dir first
    
    if args.report_directory:
        report_path = args.report_directory
    else:
        report_path = output_path
    proto = get_protocol(report_path)
    if proto == "" or proto == "file":
        output_report_dir = report_path
    elif proto == "gs":
        remote_report_path, output_report_dir = os.path.split(report_path.strip("/"))
    if os.path.exists(output_report_dir):
        shutil.rmtree(output_report_dir)
    os.mkdir(output_report_dir)
        
    logger.info(f"Writing diagnostics plots report to {report_path}")
    
    report_sections = make_all_plots(states_and_tendencies, output_report_dir)
    create_report(report_sections, "one_step_diagnostics", output_report_dir)
    
    if proto == "gs":
        copy(output_report_dir, remote_report_path)
