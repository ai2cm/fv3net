import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import gcsfs
import logging
from numpy import random
import os
import shutil
import xarray as xr

from . import helpers
from vcm import coarsen
from vcm.calc import apparent_source
from vcm.cloud import gsutil
from vcm.cubedsphere.constants import (
    COORD_Z_CENTER,
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
    VAR_LON_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_OUTER,
    VAR_LAT_OUTER,
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
    GRID_VARS
)
from vcm.cubedsphere import open_cubed_sphere
from vcm.cubedsphere.coarsen import rename_centered_xy_coords, shift_edge_var_to_center
from vcm.fv3_restarts import (
    TIME_FMT,
    open_restarts,
    _parse_forecast_dt,
    _parse_time,
    _parse_time_string,
    _set_forecast_time_coord,
)
from vcm.select import mask_to_surface_type
logger = logging.getLogger()
logger.setLevel(logging.INFO)

SAMPLE_DIM = "sample"
SAMPLE_CHUNK_SIZE = 1500

GRID_VARS = [COORD_X_CENTER, COORD_Y_CENTER, COORD_X_OUTER, COORD_Y_OUTER, "area"]
INPUT_VARS = ["sphum", "T", "delp", "u", "v", "slmsk"]
TARGET_VARS = ["Q1", "Q2", "QU", "QV"]


def run(args, pipeline_args):
    fs = gcsfs.GCSFileSystem(project=args.gcs_project)
    data_path = os.path.join(args.gcs_bucket, args.gcs_input_data_path)
    gcs_urls = ["gs://" + run_dir_path for run_dir_path in sorted(fs.ls(data_path))]
    _save_grid_spec(fs, gcs_urls[0], args.gcs_output_data_dir, args.gcs_bucket)
    data_batch_urls = _get_url_batches(gcs_urls, args.timesteps_per_output_file)
    train_test_labels = _test_train_split(
        data_batch_urls, args.train_fraction, args.random_seed
    )
    dt_forecast = _parse_forecast_dt(gcs_urls[0])

    print(f"Processing {len(data_batch_urls)} subsets...")
    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | beam.Create(data_batch_urls)
            | "LoadCloudData" >> beam.Map(_open_cloud_data, dt_forecast=dt_forecast)
            | "CreateTrainingCols" >> beam.Map(_create_train_cols)
            | "MaskToSurfaceType"
            >> beam.Map(mask_to_surface_type, surface_type=args.mask_to_surface_type)
            | "StackAndDropNan" >> beam.Map(_stack_and_drop_nan_samples)
            | "WriteToZarr"
            >> beam.Map(
                _write_remote_train_zarr,
                gcs_dest_dir=args.gcs_output_data_dir,
                bucket=args.gcs_bucket,
                train_test_labels=train_test_labels,
            )
        )


def _save_grid_spec(fs, run_dir, gcs_output_data_dir, gcs_bucket):
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
    _write_remote_train_zarr(
        grid, gcs_output_data_dir, gcs_bucket, zarr_filename="grid_spec.zarr"
    )
    logger.info(
        f"Wrote grid spec to "
        f"{os.path.join(gcs_bucket, gcs_output_data_dir, 'grid_spec.zarr')}"
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


def _test_train_split(url_batches, train_frac, random_seed=1234):
    """ Randomly assigns train/test set labels to each batch

    Args:
        url_batches: nested list where inner lists are groupings of input urls
        train_frac:
        random_seed:

    Returns:

    """
    random.seed(random_seed)
    random.shuffle(url_batches)
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


def _open_cloud_data(run_dirs, dt_forecast_sec):
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
    for run_dir in run_dirs:
        t_init = _parse_time_string(_parse_time(run_dir))
        ds_run = (
            open_restarts(run_dir)
            [INPUT_VARS]
            .expand_dims(dim={INIT_TIME_DIM: [t_init]})
        )
        ds_run = _set_forecast_time_coord(ds_run) \
            .isel({FORECAST_TIME_DIM: slice(-2, None)})
        ds_runs.append(ds_run)
    return xr.concat(ds_runs, INIT_TIME_DIM)


def _create_train_cols(ds, cols_to_keep=INPUT_VARS + TARGET_VARS):
    """

    Args:
        ds: xarray dataset, must have variables ['u', 'v', 'T', 'sphum']

    Returns:
        xarray dataset with variables in INPUT_VARS + TARGET_VARS + GRID_VARS
    """
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
        .squeeze(drop=True)
    )
    return ds


def _stack_and_drop_nan_samples(ds):
    """

    Args:
        ds: xarray dataset

    Returns:
        xr dataset stacked into sample dimension and with NaN elements dropped
         (the masked out land/sea type)
    """

    ds = (
        ds.stack({SAMPLE_DIM: [dim for dim in ds.dims if dim != COORD_Z_CENTER]})
        .transpose(SAMPLE_DIM, COORD_Z_CENTER)
        .reset_index(SAMPLE_DIM)
        .dropna(SAMPLE_DIM)
        .chunk(SAMPLE_CHUNK_SIZE, ds.sizes[COORD_Z_CENTER])
    )
    return ds


def _write_remote_train_zarr(
    ds,
    gcs_dest_dir,
    bucket="gs://vcm-ml-data",
    zarr_filename=None,
    train_test_labels=None,
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
    logger.info("Writing to zarr...")
    if not zarr_filename:
        zarr_filename = _path_from_first_timestep(ds, train_test_labels)
    output_path = os.path.join(bucket, gcs_dest_dir, zarr_filename)
    ds.to_zarr(zarr_filename, mode="w")
    gsutil.copy(zarr_filename, output_path)
    logger.info(f"Done writing zarr to {output_path}")
    shutil.rmtree(zarr_filename)


def _path_from_first_timestep(ds, train_test_labels=None):
    """ Uses first init time as zarr filename, and appends a 'train'/'test' subdir
    if a dict of labels is provided

    Args:
        ds:
        train_test_labels: optional dict with keys ["test", "train"] and values lists of
            timestep strings that go to each set

    Returns:
        path in args.gcs_output_dir to write the zarr to
    """
    timestep = min(ds[INIT_TIME_DIM].values).strftime(TIME_FMT)
    if isinstance(train_test_labels, dict):
        try:
            if timestep in train_test_labels["train"]:
                train_test_subdir = "train"
            elif timestep in train_test_labels["test"]:
                train_test_subdir = "test"
        except KeyError:
            logger.warning(
                "train_test_labels dict does not have keys ['train', 'test']."
                "Will write zarrs directly to gcs_output_dir."
            )
            train_test_subdir = ""
    else:
        logger.info(
            "No train_test_labels dict provided."
            "Will write zarrs directly to gcs_output_dir."
        )
        train_test_subdir = ""
    return os.path.join(train_test_subdir, timestep + ".zarr")


def _merge_hires_data(ds_run, diag_c384_path, diag_data_vars):
    init_times = ds_run[INIT_TIME_DIM].values
    diags_c384 = helpers._load_c384_diag(diag_c384_path, init_times, diag_data_vars)
    diags_c48 = coarsen.weighted_block_average(
        diags_c384,
        diags_c384["area_coarse"],
        x_dim = COORD_X_CENTER,
        y_dim = COORD_Y_CENTER,
        coarsening_factor=8
    ).unify_chunks()
    features_diags_c48 = helpers._coarsened_features(diags_c48)
    return xr.merge([ds_run, features_diags_c48])
