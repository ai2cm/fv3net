import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import fsspec
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
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
)
from vcm.cubedsphere import open_cubed_sphere
from vcm.cubedsphere.coarsen import rename_centered_xy_coords, shift_edge_var_to_center
from vcm.fv3_restarts import (
    open_restarts_with_time_coordinates,
    _parse_time,
    _parse_time_string,
    _split_url,
)
from vcm.select import mask_to_surface_type
from vcm.convenience import parse_timestep_from_path
from fv3net import COARSENED_DIAGS_ZARR_NAME

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SAMPLE_DIM = "sample"
SAMPLE_CHUNK_SIZE = 1500

RESTART_VARS = [
    "sphum",
    "T",
    "delp",
    "u",
    "v",
    "slmsk",
    "phis",
    "tsea",
    "slope",
    "DZ",
    "W",
]
TARGET_VARS = ["Q1", "Q2", "QU", "QV"]
HIRES_VARS = [
    "LHTFLsfc_coarse",
    "SHTFLsfc_coarse",
    "PRATEsfc_coarse",
    "DSWRFtoa_coarse",
    "DSWRFsfc_coarse",
    "USWRFtoa_coarse",
    "USWRFsfc_coarse",
    "DLWRFsfc_coarse",
    "ULWRFtoa_coarse",
    "ULWRFsfc_coarse",
]
RENAMED_HIRES_VARS = {
    "DSWRFtoa_coarse": "insolation",
    "LHTFLsfc_coarse": "LHF",
    "SHTFLsfc_coarse": "SHF",
    "PRATEsfc_coarse": "precip_sfc",
}


def run(args, pipeline_args):
    proto, path = _split_url(args.gcs_input_data_path)
    fs = fsspec.filesystem(proto)
    gcs_urls = [
        "gs://" + run_dir_path
        for run_dir_path in sorted(fs.ls(path))
        if _filter_timestep(run_dir_path)
    ]
    _save_grid_spec(fs, gcs_urls[0], args.gcs_output_data_dir)
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
            | "MaskToSurfaceType"
            >> beam.Map(
                _try_mask_to_surface_type, surface_type=args.mask_to_surface_type
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


def _save_grid_spec(fs, run_dir, gcs_output_data_dir):
    """ Reads grid spec from diag files in a run dir and writes to GCS

    Args:
        fs: GCSFileSystem object
        run_dir: run dir to read grid data from. Using the first timestep should be fine
        gcs_output_data_dir: Write path
        gcs_bucket: GCS bucket to write to

    Returns:
        None
    """
    grid_info_files = [
        "gs://" + filename
        for filename in fs.ls(run_dir)
        if "atmos_dt_atmos" in filename
    ]
    os.makedirs("temp_grid_spec", exist_ok=True)
    gsutil.copy_many(grid_info_files, "temp_grid_spec")
    grid = open_cubed_sphere(
        "temp_grid_spec/atmos_dt_atmos",
        num_subtiles=1,
        pattern="{prefix}.tile{tile:d}.nc",
    )[["area", VAR_LAT_OUTER, VAR_LON_OUTER, VAR_LAT_CENTER, VAR_LON_CENTER]]
    _write_remote_train_zarr(grid, gcs_output_data_dir, zarr_name="grid_spec.zarr")
    logger.info(
        f"Wrote grid spec to " f"{os.path.join(gcs_output_data_dir, 'grid_spec.zarr')}"
    )
    shutil.rmtree("temp_grid_spec")


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
    """ Randomly assigns train/test set labels to each batch

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
            _parse_time(batch_urls[0]) for batch_urls in url_batches[:num_train_batches]
        ],
        "test": [
            _parse_time(batch_urls[0]) for batch_urls in url_batches[num_train_batches:]
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
    try:
        logger.info(
            f"Using run dirs for batch: "
            f"{[os.path.basename(run_dir[:-1]) for run_dir in run_dirs]}"
        )
        ds_runs = []
        for run_dir in run_dirs:
            t_init = _parse_time(run_dir)
            ds_run = (
                open_restarts_with_time_coordinates(run_dir)[RESTART_VARS]
                .rename({"time": FORECAST_TIME_DIM})
                .isel({FORECAST_TIME_DIM: slice(-2, None)})
                .expand_dims(dim={INIT_TIME_DIM: [_parse_time_string(t_init)]})
            )

            ds_run = helpers._set_relative_forecast_time_coord(ds_run)
            ds_runs.append(ds_run)
        return xr.concat(ds_runs, INIT_TIME_DIM)
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to open restarts from cloud: {e}")


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
        ds["QU"] = apparent_source(ds.u)
        ds["QV"] = apparent_source(ds.v)
        ds["Q1"] = apparent_source(ds.T)
        ds["Q2"] = apparent_source(ds.sphum)
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
        diags_c48 = helpers.load_diag(full_zarr_path, init_times)[HIRES_VARS]
        features_diags_c48 = diags_c48.rename(RENAMED_HIRES_VARS)
        return xr.merge([ds_run, features_diags_c48])
    except (KeyError, AttributeError, ValueError, TypeError) as e:
        logger.error(f"Failed to merge in features from high res diagnostics: {e}")


def _try_mask_to_surface_type(ds, surface_type):
    surface_type = None if surface_type == "None" else surface_type
    try:
        return mask_to_surface_type(ds, surface_type)
    except (AttributeError, ValueError, TypeError) as e:
        logger.error(f"Failed masking to surface type: {e}")


def _write_remote_train_zarr(
    ds, gcs_output_dir, zarr_name=None, train_test_labels=None,
):
    """Writes temporary zarr on worker and moves it to GCS

    Args:
        ds: xr dataset for single training batch
        gcs_dest_path: write location on GCS
        zarr_filename: name for zarr, use first timestamp as label
        bucket: GCS bucket
        train_test_labels: optional dict with
    Returns:
        None
    """
    try:
        if not zarr_name:
            zarr_name = helpers._path_from_first_timestep(ds, train_test_labels)
        output_path = os.path.join(gcs_output_dir, zarr_name)
        ds.to_zarr(zarr_name, mode="w", consolidated=True)
        gsutil.copy(zarr_name, output_path)
        logger.info(f"Done writing zarr to {output_path}")
        shutil.rmtree(zarr_name)
    except (ValueError, AttributeError, TypeError, RuntimeError) as e:
        logger.error(f"Failed to write zarr: {e}")


def _filter_timestep(path):
    try:
        parse_timestep_from_path(path)
        return True
    except ValueError:
        return False
