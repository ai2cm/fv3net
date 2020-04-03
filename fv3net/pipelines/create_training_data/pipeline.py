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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TIME_FMT = "%Y%m%d.%H%M%S"
GRID_SPEC_FILENAME = "grid_spec.zarr"

# forecast time step used to calculate the FV3 run tendency
FORECAST_TIME_INDEX_FOR_C48_TENDENCY = 13
# forecast time step used to calculate the high res tendency
FORECAST_TIME_INDEX_FOR_HIRES_TENDENCY = FORECAST_TIME_INDEX_FOR_C48_TENDENCY


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
    ds_full = xr.open_zarr(fs.get_mapper(args.gcs_input_data_path))
    ds_full = _str_time_dim_to_datetime(ds_full, names["init_time_dim"])
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
            | "PreprocessOneStepData"
            >> beam.Map(
                _preprocess_one_step_data,
                flux_vars=names["flux_vars"],
                suffix_coarse_train=names["suffix_coarse_train"],
                step_time_dim=names["step_time_dim"],
                forecast_time_dim=names["forecast_time_dim"],
                coord_begin_step=names["coord_begin_step"],
                wind_vars=(names["var_x_wind"], names["var_y_wind"]),
                edge_to_center_dims=names["edge_to_center_dims"],
            )
            | "AddApparentSources"
            >> beam.Map(
                _add_apparent_sources,
                init_time_dim=names["init_time_dim"],
                forecast_time_dim=names["forecast_time_dim"],
                var_source_name_map=names["var_source_name_map"],
                tendency_tstep_onestep=FORECAST_TIME_INDEX_FOR_C48_TENDENCY,
                tendency_tstep_highres=FORECAST_TIME_INDEX_FOR_HIRES_TENDENCY,
            )
            | "SelectOneStepCols" >> beam.Map(lambda x: x[list(names["one_step_vars"])])
            | "MergeHiresDiagVars"
            >> beam.Map(
                _merge_hires_data,
                diag_c48_path=args.diag_c48_path,
                coarsened_diags_zarr_name=COARSENED_DIAGS_ZARR_NAME,
                flux_vars=names["flux_vars"],
                suffix_hires=names["suffix_hires"],
                init_time_dim=names["init_time_dim"],
                renamed_dims=names["renamed_dims"],
            )
            | "WriteToZarr"
            >> beam.Map(
                _write_remote_train_zarr,
                chunk_sizes=chunk_sizes,
                init_time_dim=names["init_time_dim"],
                gcs_output_dir=args.gcs_output_data_dir,
                train_test_labels=train_test_labels,
            )
        )


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
    try:
        ds = ds.sel({step_time_dim: coord_begin_step}).rename(renamed_one_step_vars)
        ds = helpers._convert_forecast_time_to_timedelta(ds, forecast_time_dim)
        # center vars located on cell edges
        for wind_var in wind_vars:
            ds[wind_var] = helpers._shift_edge_var_to_center(
                ds[wind_var], edge_to_center_dims
            )
        return ds
    except (KeyError, ValueError) as e:
        logger.error(f"Failed step PreprocessTrainData: {e}")


def _add_apparent_sources(
    ds,
    tendency_tstep_onestep,
    tendency_tstep_highres,
    init_time_dim,
    forecast_time_dim,
    var_source_name_map,
):
    try:
        for var_name, source_name in var_source_name_map.items():
            ds[source_name] = apparent_source(
                ds[var_name],
                coarse_tstep_idx=tendency_tstep_onestep,
                highres_tstep_idx=tendency_tstep_highres,
                t_dim=init_time_dim,
                s_dim=forecast_time_dim,
            )
        ds = ds.isel(
            {
                init_time_dim: slice(None, ds.sizes[init_time_dim] - 1),
                forecast_time_dim: 0,
            }
        ).drop(forecast_time_dim)
        return ds
    except (ValueError, TypeError) as e:
        logger.error(f"Failed step CreateTrainingCols: {e}")


def _merge_hires_data(
    ds_run,
    diag_c48_path,
    coarsened_diags_zarr_name,
    flux_vars,
    suffix_hires,
    init_time_dim,
    renamed_dims,
):

    renamed_high_res_vars = {
        **{
            f"{var}_coarse": f"{var}_{suffix_hires}"
            for var in flux_vars
            if var in list(ds_run.data_vars)
        },
        "LHTFLsfc_coarse": f"latent_heat_flux_{suffix_hires}",
        "SHTFLsfc_coarse": f"sensible_heat_flux_{suffix_hires}",
    }
    if not diag_c48_path:
        return ds_run
    try:
        init_times = ds_run[init_time_dim].values
        full_zarr_path = os.path.join(diag_c48_path, coarsened_diags_zarr_name)
        diags_c48 = helpers.load_hires_prog_diag(full_zarr_path, init_times)[
            list(renamed_high_res_vars.keys())
        ]
        renamed_dims = {
            dim: renamed_dims[dim] for dim in renamed_dims if dim in diags_c48.dims
        }
        features_diags_c48 = diags_c48.rename({**renamed_high_res_vars, **renamed_dims})
        return xr.merge([ds_run, features_diags_c48])
    except (KeyError, AttributeError, ValueError, TypeError) as e:
        logger.error(f"Failed to merge in features from high res diagnostics: {e}")


def _write_remote_train_zarr(
    ds,
    gcs_output_dir,
    init_time_dim,
    time_fmt=TIME_FMT,
    zarr_name=None,
    train_test_labels=None,
    chunk_sizes=None,
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
            zarr_name = helpers._path_from_first_timestep(
                ds, init_time_dim, time_fmt, train_test_labels
            )
            ds = ds.chunk(chunk_sizes)
        output_path = os.path.join(gcs_output_dir, zarr_name)
        ds.to_zarr(zarr_name, mode="w", consolidated=True)
        gsutil.copy(zarr_name, output_path)
        logger.info(f"Done writing zarr to {output_path}")
        shutil.rmtree(zarr_name)
    except (ValueError, AttributeError, TypeError, RuntimeError) as e:
        logger.error(f"Failed to write zarr: {e}")
