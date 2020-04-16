import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import os
import shutil
import xarray as xr

from . import helpers
from vcm.calc import apparent_source
from vcm.cloud import gsutil
from vcm.cloud.fsspec import get_fs
from vcm import parse_datetime_from_str
from fv3net import COARSENED_DIAGS_ZARR_NAME
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TIME_FMT = "%Y%m%d.%H%M%S"
GRID_SPEC_FILENAME = "grid_spec.zarr"
ZARR_NAME = "big.zarr"
OUTPUT_ZARR = "big.zarr"

# forecast time step used to calculate the FV3 run tendency
FORECAST_TIME_INDEX_FOR_C48_TENDENCY = 13
# forecast time step used to calculate the high res tendency
FORECAST_TIME_INDEX_FOR_HIRES_TENDENCY = FORECAST_TIME_INDEX_FOR_C48_TENDENCY

from vcm.convenience import round_time


def run(args, pipeline_args, names):
    """ Divide full one step output data into batches to be sent
    through a beam pipeline, which writes training/test data zarrs
    
    Args:
        args ([arg namespace]): for named args in the main function
        pipeline_args ([arg namespace]): additional args for the pipeline
        names ([dict]): Contains information related to the variable
            and dimension names from the one step output and created
            by the pipeline.
    """
    fs = get_fs(args.gcs_input_data_path)
    ds_full = xr.open_zarr(
        fs.get_mapper(os.path.join(args.gcs_input_data_path, ZARR_NAME))
    )

    init_time_dim = names["init_time_dim"]
    ds_full = _str_time_dim_to_datetime(ds_full, init_time_dim)

    # one_step operations
    # these are all lazy
    ds = _add_physics_tendencies(
        ds_full,
        physics_tendency_names=names["physics_tendency_names"],
        forecast_time_dim=names["forecast_time_dim"],
        step_dim=names["step_time_dim"],
        coord_before_physics=names["coord_before_physics"],
        coord_after_physics=names["coord_after_physics"],
    )
    ds = _preprocess_one_step_data(
        ds,
        flux_vars=names["diag_vars"],
        suffix_coarse_train=names["suffix_coarse_train"],
        step_time_dim=names["step_time_dim"],
        forecast_time_dim=names["forecast_time_dim"],
        coord_begin_step=names["coord_begin_step"],
        wind_vars=(names["var_x_wind"], names["var_y_wind"]),
        edge_to_center_dims=names["edge_to_center_dims"],
    )

    ds = _add_apparent_sources(
        ds,
        init_time_dim=init_time_dim,
        forecast_time_dim=names["forecast_time_dim"],
        var_source_name_map=names["var_source_name_map"],
        tendency_tstep_onestep=FORECAST_TIME_INDEX_FOR_C48_TENDENCY,
        tendency_tstep_highres=FORECAST_TIME_INDEX_FOR_HIRES_TENDENCY,
    )

    ds = ds[list(names["one_step_vars"])]
    ds = ds.rename({"initial_time": "time"})

    # load high-res-data
    # TODO fix the hard code
    high_res_mapper = fs.get_mapper(
        args.diag_c48_path + "/gfsphysics_15min_coarse.zarr"
    )
    high_res = xr.open_zarr(high_res_mapper, consolidated=True)
    times_rounded = [round_time(time) for time in high_res.time.values]
    high_res_rounded_times = high_res.assign_coords(time=times_rounded)

    hr_rename = names["high_res"]
    high_res_renamed = high_res_rounded_times[list(hr_rename)].rename(hr_rename)

    # merge
    merged = xr.merge([high_res_renamed, ds]).chunk(names['output_chunks'])
    _remove_encoding(merged)

    # save out
    mapper = fs.get_mapper(os.path.join(args.gcs_output_data_dir, OUTPUT_ZARR))
    merged.to_zarr(mapper, mode="w")


def _remove_encoding(ds):
    ds.encoding = {}
    for variable in ds:
        ds[variable].encoding = {}


def _str_time_dim_to_datetime(ds, time_dim):
    datetime_coords = [
        parse_datetime_from_str(time_str) for time_str in ds[time_dim].values
    ]
    return ds.assign_coords({time_dim: datetime_coords})


def _divide_data_batches(
    ds_full, timesteps_per_output_file, train_fraction, init_time_dim
):
    """ Divides the full dataset of one step run outputs into batches designated
    for training or test sets.
    Args:
        ds_full (xr dataset): dataset read in from the big zarr of all one step outputs
        timesteps_per_output_file (int): number of timesteps to save per train batch
        train_fraction (float): fraction of initial timesteps to use as training
    
    Returns:
        tuple (
            list of datasets selected to the timesteps for each output,
            dict of train and test timesteps
            )
    """
    timestep_batches = _get_timestep_batches(
        ds_full, timesteps_per_output_file, init_time_dim
    )
    train_test_labels = _test_train_split(timestep_batches, train_fraction)
    timestep_batches_reordered = _reorder_batches(timestep_batches, train_fraction)
    data_batches = [
        ds_full.sel({init_time_dim: timesteps})
        for timesteps in timestep_batches_reordered
    ]
    return (data_batches, train_test_labels)


def _reorder_batches(sorted_batches, train_frac):
    """Uniformly distribute the test batches within the list of batches to run,
    so that they are not all left to the end of the job. This is so that we don't
    have to run a training data job to completion in order to get the desired
    train/test ratio.

    Args:
        sorted_batches (nested list):of run dirs per batch
        train_frac (float): fraction of batches for use in training

    Returns:
        nested list of batch urls, reordered so that test times are uniformly
        distributed in list
    """
    num_batches = len(sorted_batches)
    split_index = int(train_frac * num_batches)
    train_set = sorted_batches[:split_index]
    test_set = sorted_batches[split_index:]
    train_test_ratio = int(train_frac / (1 - train_frac))
    reordered_batches = []
    while len(train_set) > 0:
        if len(test_set) > 0:
            reordered_batches.append(test_set.pop(0))
        for i in range(train_test_ratio):
            if len(train_set) > 0:
                reordered_batches.append(train_set.pop(0))
    return reordered_batches


def _save_grid_spec(
    ds, gcs_output_data_dir, grid_vars, grid_spec_filename, init_time_dim
):
    """ Reads grid spec from diag files in a run dir and writes to GCS

    Args:
        fs: GCSFileSystem object
        run_dir: run dir to read grid data from. Using the first timestep should be fine
        gcs_output_data_dir: Write path

    Returns:
        None
    """
    grid = ds.isel({init_time_dim: 0})[grid_vars]
    _write_remote_train_zarr(
        grid, gcs_output_data_dir, init_time_dim, zarr_name=grid_spec_filename
    )
    logger.info(
        f"Wrote grid spec to "
        f"{os.path.join(gcs_output_data_dir, grid_spec_filename)}"
    )
    return


def _get_timestep_batches(ds, timesteps_per_output_file, init_time_dim):
    """ Groups initalization timesteps into lists of max length
    (args.timesteps_per_output_file + 1). The last file in each grouping is only
    used to calculate the hi res tendency, and is dropped from the final
    batch training zarr.

    Args:
        gcs_urls: list of urls to be grouped into batches
        timesteps_per_output_file: number of initialization timesteps that will be in
        each final train dataset batch
    Returns:
        nested list where inner lists are groupings of timesteps
    """
    timesteps = sorted(ds[init_time_dim].values)
    num_outputs = (len(timesteps) - 1) // timesteps_per_output_file
    timestep_batches = []
    for i in range(num_outputs):
        start_ind = timesteps_per_output_file * i
        stop_ind = timesteps_per_output_file * i + (timesteps_per_output_file + 1)
        timestep_batches.append(timesteps[start_ind:stop_ind])
    num_leftover = len(timesteps) % timesteps_per_output_file
    remainder_urls = [timesteps[-num_leftover:]] if num_leftover > 1 else []
    timestep_batches += remainder_urls
    return timestep_batches


def _test_train_split(timestep_batches, train_frac):
    """ Assigns train/test set labels to each batch, split by init timestamp

    Args:
        url_batches: nested list where inner lists are groupings of input urls,
        ordered by time
        train_frac: Float [0, 1]

    Returns:
        dict lookup for each batch's set to save to
    """
    if train_frac > 1:
        train_frac = 1
        logger.warning("Train fraction provided > 1. Will set to 1.")
    num_train_batches = int(len(timestep_batches) * train_frac)
    labels = {
        "train": [
            timesteps[0].strftime(TIME_FMT)
            for timesteps in timestep_batches[:num_train_batches]
        ],
        "test": [
            timesteps[0].strftime(TIME_FMT)
            for timesteps in timestep_batches[num_train_batches:]
        ],
    }
    return labels


def _add_physics_tendencies(
    ds,
    physics_tendency_names,
    forecast_time_dim,
    step_dim,
    coord_before_physics,
    coord_after_physics,
):
    # physics timestep is same as forecast time step [s]
    dt = ds[forecast_time_dim].values[1]
    for var, tendency in physics_tendency_names.items():
        ds[tendency] = (
            ds[var].sel({step_dim: coord_after_physics})
            - ds[var].sel({step_dim: coord_before_physics})
        ) / dt

    return ds


def _preprocess_one_step_data(
    ds,
    flux_vars,
    suffix_coarse_train,
    forecast_time_dim,
    step_time_dim,
    coord_begin_step,
    wind_vars,
    edge_to_center_dims,
):
    renamed_one_step_vars = {
        var: f"{var}_{suffix_coarse_train}"
        for var in flux_vars
        if var in list(ds.data_vars)
    }
    ds = ds.sel({step_time_dim: coord_begin_step}).drop(step_time_dim)
    ds = ds.rename(renamed_one_step_vars)
    ds = helpers._convert_forecast_time_to_timedelta(ds, forecast_time_dim)
    # center vars located on cell edges
    for wind_var in wind_vars:
        ds[wind_var] = helpers._shift_edge_var_to_center(
            ds[wind_var], edge_to_center_dims
        )
    return ds


def _add_apparent_sources(
    ds,
    tendency_tstep_onestep,
    tendency_tstep_highres,
    init_time_dim,
    forecast_time_dim,
    var_source_name_map,
):
    for var_name, source_name in var_source_name_map.items():
        ds[source_name] = apparent_source(
            ds[var_name],
            coarse_tstep_idx=tendency_tstep_onestep,
            highres_tstep_idx=tendency_tstep_highres,
            t_dim=init_time_dim,
            s_dim=forecast_time_dim,
        )
    ds = ds.isel(
        {init_time_dim: slice(None, ds.sizes[init_time_dim] - 1), forecast_time_dim: 0,}
    ).drop(forecast_time_dim)
    return ds