import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import os
import shutil
import xarray as xr

from . import helpers
from . import names
from vcm.calc import apparent_source
from vcm.cloud import gsutil
from vcm.cloud.fsspec import get_fs
from vcm import parse_datetime_from_str
from fv3net import COARSENED_DIAGS_ZARR_NAME

logger = logging.getLogger()
logger.setLevel(logging.INFO)

GRID_SPEC_FILENAME = "grid_spec.zarr"

# forecast time step used to calculate the FV3 run tendency
FORECAST_TIME_INDEX_FOR_C48_TENDENCY = 14
# forecast time step used to calculate the high res tendency
FORECAST_TIME_INDEX_FOR_HIRES_TENDENCY = FORECAST_TIME_INDEX_FOR_C48_TENDENCY


def run(args, pipeline_args, names):
    fs = get_fs(args.gcs_input_data_path)
    ds_full = xr.open_zarr(fs.get_mapper(args.gcs_input_data_path))
    _save_grid_spec(
        ds_full,
        args.gcs_output_data_dir,
        grid_vars=names["grid_vars"],
        grid_spec_filename=GRID_SPEC_FILENAME,
        init_time_dim=names["init_time_dim"],
    )
    data_batches, train_test_labels = _divide_data_batches(
        ds_full,
        args.timesteps_per_output_file,
        args.train_fraction,
        init_time_dim=names["init_time_dim"],
    )
    chunk_sizes = {
        "tile": 1,
        names["init_time_dim"]: 1,
        names["coord_y_center"]: 24,
        names["coord_x_center"]: 24,
        names["coord_z_center"]: 79,
    }
    logger.info(f"Processing {len(data_batches)} subsets...")
    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | beam.Create(data_batches)
            | "CreateTrainingCols"
            >> beam.Map(
                _create_train_cols,
                cols_to_keep=names["one_step_vars"] + names["target_vars"],
                init_time_dim=names["init_time_dim"],
                step_time_dim=names["step_time_dim"],
                forecast_time_dim=names["forecast_time_dim'"],
                coord_begin_step=names["coord_begin_step"],
                var_source_name_map=names["var_source_name_map"],
                forecast_timestep_for_onestep=FORECAST_TIME_INDEX_FOR_C48_TENDENCY,
                forecast_timestep_for_highres=FORECAST_TIME_INDEX_FOR_HIRES_TENDENCY,
            )
            | "MergeHiresDiagVars"
            >> beam.Map(
                _merge_hires_data,
                diag_c48_path=args.diag_c48_path,
                coarsened_diags_zarr_name=COARSENED_DIAGS_ZARR_NAME,
                renamed_high_res_vars=names["renamed_high_res_vars"],
                init_time_dim=names["init_time_dim"],
            )
            | "WriteToZarr"
            >> beam.Map(
                _write_remote_train_zarr,
                gcs_output_dir=args.gcs_output_data_dir,
                train_test_labels=train_test_labels,
                chunk_sizes=chunk_sizes,
            )
        )


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
    init_datetime_coords = [
        parse_datetime_from_str(init_time)
        for init_time in ds_full[init_time_dim].values
    ]
    ds_full = ds_full.assign_coords({init_time_dim: init_datetime_coords})
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
    _write_remote_train_zarr(grid, gcs_output_data_dir, zarr_name=grid_spec_filename)
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
        "train": [timesteps[0] for timesteps in timestep_batches[:num_train_batches]],
        "test": [timesteps[0] for timesteps in timestep_batches[num_train_batches:]],
    }
    return labels


def _open_cloud_data(ds, forecast_time_dim, step_time_dim, coord_begin_step):
    """Selects forecast time dimension

    Args:
        fs: GCSFileSystem
        run_dirs: list of GCS urls to open

    Returns:
        xarray dataset of concatenated zarrs in url list
    """
    logger.info(f"Using timesteps for batch: {ds[names.init_time_dim].values}.")
    ds = ds.isel({forecast_time_dim: [0, -2, -1]}).sel(
        {step_time_dim: coord_begin_step}
    )
    return ds


def _create_train_cols(
    ds,
    forecast_timestep_for_onestep,
    forecast_timestep_for_highres,
    init_time_dim,
    forecast_time_dim,
    step_time_dim,
    coord_begin_step,
    var_source_name_map,
    cols_to_keep=names["one_step_vars"] + names["target_vars"],
):
    """ Calculate apparent sources for target variables and keep feature vars
    
    Args:
        ds (xarray dataset): must have the specified feature vars in cols_to_keep
            as well as vars needed to calculate apparent sources
        tendency_forecast_time_index (int): index of the forecast time step to use
            when calculating apparent source
        init_time_dim (str): name of initial time dimension
        forecast_time_dim (str): name of forecast time dimension
        step_time_dim (str): name of step time dimension
        coord_begin_step (str): coordinate for the first step in each forecast timestep
            (inital state before dynamics or physics are stepped)
        var_source_name_map (dict): dict that maps variable name to its apparent source
            variable name
        cols_to_keep ([str], optional): List of variable names that will be in final
            train dataset. Defaults to names.one_step_vars+names.target_vars.
    
    Returns:
        xarray dataset containing feature and target data variables
    """

    try:
        ds = ds.sel({step_time_dim: coord_begin_step})
        for var_name, source_name in var_source_name_map.items():
            ds[source_name] = apparent_source(
                ds[var_name],
                forecast_timestep_for_onestep,
                forecast_timestep_for_highres,
                t_dim=init_time_dim,
                s_dim=forecast_time_dim,
            )
        ds = (
            ds[cols_to_keep]
            .isel(
                {
                    init_time_dim: slice(None, ds.sizes[init_time_dim] - 1),
                    forecast_time_dim: 0,
                }
            )
            .drop(forecast_time_dim)
        )
        return ds
    except (ValueError, TypeError) as e:
        logger.error(f"Failed step CreateTrainingCols: {e}")


def _merge_hires_data(
    ds_run,
    diag_c48_path,
    coarsened_diags_zarr_name,
    renamed_high_res_vars,
    init_time_dim,
):
    if not diag_c48_path:
        return ds_run
    try:
        init_times = ds_run[init_time_dim].values
        full_zarr_path = os.path.join(diag_c48_path, coarsened_diags_zarr_name)
        diags_c48 = helpers.load_hires_prog_diag(full_zarr_path, init_times)[
            list(renamed_high_res_vars.keys())
        ]
        features_diags_c48 = diags_c48.rename(renamed_high_res_vars)
        return xr.merge([ds_run, features_diags_c48])
    except (KeyError, AttributeError, ValueError, TypeError) as e:
        logger.error(f"Failed to merge in features from high res diagnostics: {e}")


def _write_remote_train_zarr(
    ds, gcs_output_dir, chunk_sizes, zarr_name=None, train_test_labels=None
):
    """Writes temporary zarr on worker and moves it to GCS

    Args:
        ds: xr dataset for single training batch
        gcs_dest_path: write location on GCS
        zarr_filename: name for zarr, use first timestamp as label
        train_test_labels: optional dict with
    Returns:
        None
    """
    try:
        if not zarr_name:
            zarr_name = helpers._path_from_first_timestep(ds, train_test_labels)
            ds = ds.chunk(chunk_sizes)
        output_path = os.path.join(gcs_output_dir, zarr_name)
        ds.to_zarr(zarr_name, mode="w", consolidated=True)
        gsutil.copy(zarr_name, output_path)
        logger.info(f"Done writing zarr to {output_path}")
        shutil.rmtree(zarr_name)
    except (ValueError, AttributeError, TypeError, RuntimeError) as e:
        logger.error(f"Failed to write zarr: {e}")
