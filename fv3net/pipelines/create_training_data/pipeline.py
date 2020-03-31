import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import os
import shutil
import xarray as xr
import yaml

from . import helpers
from vcm.calc import apparent_source
from vcm.cloud import gsutil
from vcm.cloud.fsspec import get_fs
from vcm.cubedsphere.coarsen import rename_centered_xy_coords, shift_edge_var_to_center
from vcm.fv3_restarts import open_restarts_with_time_coordinates
from vcm import parse_timestep_str_from_path, parse_datetime_from_str
from fv3net import COARSENED_DIAGS_ZARR_NAME

logger = logging.getLogger()
logger.setLevel(logging.INFO)

INIT_TIME_DIM = "initial_time"
FORECAST_TIME_DIM = "forecast_time"
STEP_TIME_DIM = "step"
COORD_END_STEP = "after_physics"
VAR_LON_CENTER, VAR_LAT_CENTER, VAR_LON_OUTER, VAR_LAT_OUTER = (
    "lon",
    "lat",
    "lonb",
    "latb",
)
COORD_X_CENTER, COORD_Y_CENTER, COORD_Z_CENTER = ("x", "y", "z")
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
SUFFIX_HIRES = "prog"
SUFFIX_COARSE_TRAIN = "train"

VAR_X_WIND, VAR_Y_WIND = ("x_wind", "y_wind")
VAR_TEMP, VAR_SPHUM = ("air_temperature", "specific_humidity")
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
    **{f"{var}_coarse": f"{var}_{SUFFIX_HIRES}" for var in RADIATION_VARS},
    **{
        "LHTFLsfc_coarse": f"latent_heat_flux_{SUFFIX_HIRES}",
        "SHTFLsfc_coarse": f"sensible_heat_flux_{SUFFIX_HIRES}",
    },
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
    VAR_TEMP,
    VAR_SPHUM,
    VAR_X_WIND,
    VAR_Y_WIND,
]
RENAMED_ONE_STEP_VARS = {var: f"{var}_{SUFFIX_COARSE_TRAIN}" for var in RADIATION_VARS}
RENAMED_DIMS = {
    "grid_xt": "x",
    "grid_yt": "y",
    "grid_x": "x_interface",
    "grid_y": "y_interface",
}


def run(args, pipeline_args):
    logger.error(f"{args}")
    if args.var_names_yaml:
        _use_var_names_from_file(args.var_names_yaml, SUFFIX_HIRES, SUFFIX_COARSE_TRAIN)
    fs = get_fs(args.gcs_input_data_path)
    ds_full = xr.open_zarr(fs.get_mapper(args.gcs_input_data_path))
    _save_grid_spec(ds_full, args.gcs_output_data_dir)
    timestep_batches = _get_timestep_batches(ds_full, args.timesteps_per_output_file)
    train_test_labels = _test_train_split(timestep_batches, args.train_fraction)
    timestep_batches_reordered = _reorder_batches(timestep_batches, args.train_fraction)
    data_batches = [
        ds_full.sel({INIT_TIME_DIM: timesteps})
        for timesteps in timestep_batches_reordered
    ]

    logger.info(f"Processing {len(timestep_batches)} subsets...")
    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | beam.Create(data_batches)
            | "LoadCloudData" >> beam.Map(_open_cloud_data)
            | "CreateTrainingCols" >> beam.Map(_create_train_cols)
            | "MergeHiresDiagVars"
            >> beam.Map(_merge_hires_data, diag_c48_path=args.diag_c48_path)
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


def _save_grid_spec(ds, gcs_output_data_dir):
    """ Reads grid spec from diag files in a run dir and writes to GCS

    Args:
        fs: GCSFileSystem object
        run_dir: run dir to read grid data from. Using the first timestep should be fine
        gcs_output_data_dir: Write path

    Returns:
        None
    """
    grid = ds.isel({INIT_TIME_DIM: 0})[
        ["area", VAR_LAT_OUTER, VAR_LON_OUTER, VAR_LAT_CENTER, VAR_LON_CENTER]
    ]
    _write_remote_train_zarr(grid, gcs_output_data_dir, zarr_name="grid_spec.zarr")
    logger.info(
        f"Wrote grid spec to " f"{os.path.join(gcs_output_data_dir, 'grid_spec.zarr')}"
    )
    return


def _get_timestep_batches(ds, timesteps_per_output_file):
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
    timesteps = sorted(ds[INIT_TIME_DIM].values)
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


def _open_cloud_data(ds):
    """Opens multiple run directories into a single dataset, where the init time
    of each run dir is the INIT_TIME_DIM and the times within

    Args:
        fs: GCSFileSystem
        run_dirs: list of GCS urls to open

    Returns:
        xarray dataset of concatenated zarrs in url list
    """
    logger.info(f"Using timesteps for batch: {ds[INIT_TIME_DIM].values}.")
    init_datetime_coords = [
        parse_datetime_from_str(init_time) for init_time in ds[INIT_TIME_DIM].values
    ]
    ds = ds.sel(
        {FORECAST_TIME_DIM: slice(-2, None), STEP_TIME_DIM: COORD_END_STEP}
    ).assign_coords({INIT_TIME_DIM: init_datetime_coords})
    return ds


def _create_train_cols(ds):
    """

    Args:
        ds: xarray dataset, must have variables ['u', 'v', 'T', 'sphum']

    Returns:
        xarray dataset with variables in RESTART_VARS + TARGET_VARS + GRID_VARS
    """
    try:
        ds[VAR_Q_U_WIND_ML] = apparent_source(
            ds[VAR_X_WIND], t_dim=INIT_TIME_DIM, s_dim=FORECAST_TIME_DIM
        )
        ds[VAR_Q_V_WIND_ML] = apparent_source(
            ds[VAR_X_WIND], t_dim=INIT_TIME_DIM, s_dim=FORECAST_TIME_DIM
        )
        ds[VAR_Q_HEATING_ML] = apparent_source(
            ds[VAR_TEMP], t_dim=INIT_TIME_DIM, s_dim=FORECAST_TIME_DIM
        )
        ds[VAR_Q_MOISTENING_ML] = apparent_source(
            ds[VAR_SPHUM], t_dim=INIT_TIME_DIM, s_dim=FORECAST_TIME_DIM
        )
        ds = (
            ds[ONE_STEP_VARS + TARGET_VARS]
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


def _use_var_names_from_file(var_names_yaml, suffix_hires, suffix_coarse_train):
    with open(var_names_yaml, "r") as stream:
        try:
            var_names = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")
    global INIT_TIME_DIM, FORECAST_TIME_DIM, STEP_TIME_DIM, COORD_END_STEP
    INIT_TIME_DIM = var_names["initial_time_dim"]
    FORECAST_TIME_DIM = var_names["forecast_time_dim"]
    STEP_TIME_DIM = var_names["step_time_dim"]
    COORD_END_STEP = var_names["end_step_coord"]

    global COORD_X_CENTER, COORD_Y_CENTER, COORD_Z_CENTER, RENAMED_DIMS
    COORD_X_CENTER = var_names["x_coord"]
    COORD_Y_CENTER = var_names["y_coord"]
    COORD_Z_CENTER = var_names["z_coord"]
    RENAMED_DIMS = var_names["grid_dim_renaming"]

    global VAR_X_WIND, VAR_Y_WIND, VAR_TEMP, VAR_SPHUM
    VAR_X_WIND = var_names["x_wind_var"]
    VAR_Y_WIND = var_names["y_wind_var"]
    VAR_TEMP = var_names["temperature_var"]
    VAR_SPHUM = var_names["specific_humidity_var"]

    global RADIATION_VARS, ONE_STEP_VARS
    RADIATION_VARS = var_names["radiation_variables"]
    ONE_STEP_VARS = var_names["one_step_data_variables"] + [
        VAR_X_WIND,
        VAR_Y_WIND,
        VAR_TEMP,
        VAR_SPHUM,
    ]

    global RENAMED_HIGH_RES_VARS, RENAMED_ONE_STEP_VARS
    RENAMED_HIGH_RES_VARS = {
        **{f"{var}_coarse": f"{var}_{suffix_hires}" for var in RADIATION_VARS},
        **{
            "LHTFLsfc_coarse": f"latent_heat_flux_{suffix_hires}",
            "SHTFLsfc_coarse": f"sensible_heat_flux_{suffix_hires}",
        },
    }
    RENAMED_ONE_STEP_VARS = {
        var: f"{var}_{suffix_coarse_train}" for var in RADIATION_VARS
    }
