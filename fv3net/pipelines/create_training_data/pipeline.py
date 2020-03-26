import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import os
import shutil
import xarray as xr

from . import helpers
from vcm.calc import apparent_source
from vcm.cloud import gsutil
from vcm.cubedsphere.constants import (
    VAR_LON_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_OUTER,
    VAR_LAT_OUTER,
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_Z_CENTER,
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
)
from vcm.cloud import fsspec
from vcm.cubedsphere.coarsen import rename_centered_xy_coords, shift_edge_var_to_center
from vcm.fv3_restarts import open_restarts_with_time_coordinates, open_diagnostic
from vcm import parse_timestep_str_from_path, parse_datetime_from_str
from fv3net import COARSENED_DIAGS_ZARR_NAME

logger = logging.getLogger()
logger.setLevel(logging.INFO)


_CHUNK_SIZES = {
    "tile": 1,
    INIT_TIME_DIM: 1,
    COORD_Y_CENTER: 24,
    COORD_X_CENTER: 24,
    COORD_Z_CENTER: 79,
}

# residuals that the ML is training on
# high resolution tendency - coarse res model's one step tendency
VAR_Q_HEATING_ML = "dQ1"
VAR_Q_MOISTENING_ML = "dQ2"
VAR_Q_U_WIND_ML = "dQU"
VAR_Q_V_WIND_ML = "dQV"
TARGET_VARS = [VAR_Q_HEATING_ML, VAR_Q_MOISTENING_ML, VAR_Q_U_WIND_ML, VAR_Q_V_WIND_ML]

# suffixes denote whether diagnostic variable is from the coarsened
# high resolution prognostic run or the coarse res one step train data run
SUFFIX_HIRES_DIAG = "prog"
SUFFIX_COARSE_TRAIN_DIAG = "train"

RADIATION_VARS = [
   "DSWRFtoa",
   "DSWRFsfc",
   "USWRFtoa",
   "USWRFsfc",
   "DLWRFsfc",
   "ULWRFtoa",
   "ULWRFsfc",
]
RENAMED_HIGH_RES_VARS = {
    **{f"{var}_coarse": f"{var}_prog" for var in RADIATION_VARS},
    **{"LHTFLsfc_coarse": "latent_heat_flux_prog",
        "SHTFLsfc_coarse": "sensible_heat_flux_prog"}
}

ONE_STEP_VARS = RADIATION_VARS + [
   "total_precipitation",
   "surface_temperature",
   "land_sea_mask",
   "latent_heat_flux",
   "sensible_heat_flux",
   "mean_cos_zenith_angle",
   "surface_geopotential",
   "vertical_thickness_of_atmospheric_layer",
   "vertical_wind",
   "pressure_thickness_of_atmospheric_layer",
   "specific_humidity",
   "air_temperature",
   "x_wind",
   "y_wind",
]
RENAMED_ONE_STEP_VARS = {var: f"{var}_train" for var in RADIATION_VARS}


def run(args, pipeline_args):
    fs = fsspec.get_fs(args.gcs_input_data_path)
    gcs_urls = [
        "gs://" + run_dir_path
        for run_dir_path in sorted(fs.ls(args.gcs_input_data_path))
        if _filter_timestep(run_dir_path)
    ]
    _save_grid_spec(fs, gcs_urls, args.gcs_output_data_dir)
    data_batch_urls = _get_url_batches(gcs_urls, args.timesteps_per_output_file)
    train_test_labels = _test_train_split(data_batch_urls, args.train_fraction)
    data_batch_urls_reordered = _reorder_batches(data_batch_urls, args.train_fraction)

    logger.info(f"Processing {len(data_batch_urls)} subsets...")
    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | beam.Create(data_batch_urls_reordered)
            | "LoadCloudData" >> beam.Map(_open_cloud_data)
            | "CreateTrainingCols" >> beam.Map(_create_train_cols)
            | "MergeHiresDiagVars"
            >> beam.Map(_merge_hires_data, diag_c48_path=args.diag_c48_path)
            | "MergeOneStepDiagVars"
            >> beam.Map(
                _merge_onestep_diag_data, top_level_data_dir=args.gcs_input_data_path
            )
            | "WriteToZarr"
            >> beam.Map(
                _write_remote_train_zarr,
                gcs_output_dir=args.gcs_output_data_dir,
                train_test_labels=train_test_labels,
            )
        )


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


def _save_grid_spec(fs, run_dirs, gcs_output_data_dir, max_attempts=25):
    """ Reads grid spec from diag files in a run dir and writes to GCS

    Args:
        fs: GCSFileSystem object
        run_dir: run dir to read grid data from. Using the first timestep should be fine
        gcs_output_data_dir: Write path

    Returns:
        None
    """
    attempt = 0
    while attempt <= max_attempts:
        run_dir = run_dirs[attempt]
        try:
            grid = open_diagnostic(run_dir, "atmos_dt_atmos").isel(time=0)[
                ["area", VAR_LAT_OUTER, VAR_LON_OUTER, VAR_LAT_CENTER, VAR_LON_CENTER]
            ]
            _write_remote_train_zarr(
                grid, gcs_output_data_dir, zarr_name="grid_spec.zarr"
            )
            logger.info(
                f"Wrote grid spec to "
                f"{os.path.join(gcs_output_data_dir, 'grid_spec.zarr')}"
            )
            return
        except FileNotFoundError as e:
            logger.error(e)
            attempt += 1
    raise FileNotFoundError(
        f"Unable to open diag files for creating grid spec, \
        reached max attempts {max_attempts}"
    )


def _get_url_batches(gcs_urls, timesteps_per_output_file):
    """ Groups the time ordered urls into lists of max length
    (args.timesteps_per_output_file + 1). The last file in each grouping is only
    used to calculate the hi res tendency, and is dropped from the final
    batch training zarr.

    Args:
        gcs_urls: list of urls to be grouped into batches
        timesteps_per_output_file: number of initialization timesteps that will be in
        each final train dataset batch
    Returns:
        nested list where inner lists are groupings of input urls
    """
    num_outputs = (len(gcs_urls) - 1) // timesteps_per_output_file
    data_urls = []
    for i in range(num_outputs):
        start_ind = timesteps_per_output_file * i
        stop_ind = timesteps_per_output_file * i + (timesteps_per_output_file + 1)
        data_urls.append(gcs_urls[start_ind:stop_ind])
    num_leftover = len(gcs_urls) % timesteps_per_output_file
    remainder_urls = [gcs_urls[-num_leftover:]] if num_leftover > 1 else []
    data_urls += remainder_urls
    return data_urls


def _test_train_split(url_batches, train_frac):
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
    num_train_batches = int(len(url_batches) * train_frac)
    labels = {
        "train": [
            parse_timestep_str_from_path(batch_urls[0])
            for batch_urls in url_batches[:num_train_batches]
        ],
        "test": [
            parse_timestep_str_from_path(batch_urls[0])
            for batch_urls in url_batches[num_train_batches:]
        ],
    }
    return labels


def _open_cloud_data(run_dirs):
    """Opens multiple run directories into a single dataset, where the init time
    of each run dir is the INIT_TIME_DIM and the times within

    Args:
        fs: GCSFileSystem
        run_dirs: list of GCS urls to open

    Returns:
        xarray dataset of concatenated zarrs in url list
    """

    logger.info(
        f"Using run dirs for batch: "
        f"{[os.path.basename(run_dir[:-1]) for run_dir in run_dirs]}"
    )
    ds_runs = []
    try:
        for run_dir in run_dirs:
            t_init = parse_timestep_str_from_path(run_dir)
            ds_run = (
                open_restarts_with_time_coordinates(run_dir)[RESTART_VARS]
                .rename({"time": FORECAST_TIME_DIM})
                .isel({FORECAST_TIME_DIM: slice(-2, None)})
                .expand_dims(dim={INIT_TIME_DIM: [parse_datetime_from_str(t_init)]})
            )
            ds_run = helpers._set_relative_forecast_time_coord(ds_run)
            ds_runs.append(ds_run)
        return xr.concat(ds_runs, INIT_TIME_DIM)
    except (IndexError, ValueError, TypeError, AttributeError, KeyError) as e:
        logger.error(f"Failed to open restarts from cloud for rundirs {run_dir}: {e}")


def _create_train_cols(ds, cols_to_keep=RESTART_VARS + TARGET_VARS):
    """

    Args:
        ds: xarray dataset, must have variables ['u', 'v', 'T', 'sphum']

    Returns:
        xarray dataset with variables in RESTART_VARS + TARGET_VARS + GRID_VARS
    """
    try:
        da_centered_u = rename_centered_xy_coords(shift_edge_var_to_center(ds["u"]))
        da_centered_v = rename_centered_xy_coords(shift_edge_var_to_center(ds["v"]))
        ds["u"] = da_centered_u
        ds["v"] = da_centered_v
        ds[VAR_Q_U_WIND_ML] = apparent_source(ds.u)
        ds[VAR_Q_V_WIND_ML] = apparent_source(ds.v)
        ds[VAR_Q_HEATING_ML] = apparent_source(ds.T)
        ds[VAR_Q_MOISTENING_ML] = apparent_source(ds.sphum)
        ds = (
            ds[cols_to_keep]
            .isel(
                {
                    INIT_TIME_DIM: slice(None, ds.sizes[INIT_TIME_DIM] - 1),
                    FORECAST_TIME_DIM: 0,
                }
            )
            .drop(FORECAST_TIME_DIM)
        )
        if "file_prefix" in ds.coords:
            ds = ds.drop("file_prefix")
        return ds
    except (ValueError, TypeError) as e:
        logger.error(f"Failed step CreateTrainingCols: {e}")


def _merge_hires_data(ds_run, diag_c48_path):
    if not diag_c48_path:
        return ds_run
    try:
        init_times = ds_run[INIT_TIME_DIM].values
        full_zarr_path = os.path.join(diag_c48_path, COARSENED_DIAGS_ZARR_NAME)
        diags_c48 = helpers.load_hires_prog_diag(full_zarr_path, init_times)[
            list(RENAMED_HIGH_RES_VARS.keys())
        ]
        features_diags_c48 = diags_c48.rename(RENAMED_HIGH_RES_VARS)
        return xr.merge([ds_run, features_diags_c48])
    except (KeyError, AttributeError, ValueError, TypeError) as e:
        logger.error(f"Failed to merge in features from high res diagnostics: {e}")


def _merge_onestep_diag_data(ds_run, top_level_data_dir):
    try:
        init_times = ds_run[INIT_TIME_DIM].values
        diags_onestep = helpers.load_train_diag(top_level_data_dir, init_times)[
            DIAG_VARS
        ]
        diags_onestep = diags_onestep.rename(RENAMED_ONE_STEP_VARS)
        return xr.merge([ds_run, diags_onestep])

    except (IndexError, FileNotFoundError, ValueError, TypeError) as e:
        logger.error(f"Failed to merge one step run diag files: {e}")


def _write_remote_train_zarr(
    ds, gcs_output_dir, zarr_name=None, train_test_labels=None
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
            ds = ds.chunk(_CHUNK_SIZES)
        output_path = os.path.join(gcs_output_dir, zarr_name)
        ds.to_zarr(zarr_name, mode="w", consolidated=True)
        gsutil.copy(zarr_name, output_path)
        logger.info(f"Done writing zarr to {output_path}")
        shutil.rmtree(zarr_name)
    except (ValueError, AttributeError, TypeError, RuntimeError) as e:
        logger.error(f"Failed to write zarr: {e}")


def _filter_timestep(path):
    try:
        parse_timestep_str_from_path(path)
        return True
    except ValueError:
        return False
