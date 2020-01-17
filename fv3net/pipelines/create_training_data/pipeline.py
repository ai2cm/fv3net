import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from datetime import timedelta
import gcsfs
import logging
import os
import shutil
import xarray as xr

from vcm.calc import apparent_source
from vcm.cloud import gsutil
from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
    VAR_LON_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_OUTER ,
    VAR_LAT_OUTER)
from vcm.cubedsphere import open_cubed_sphere
from vcm.cubedsphere.coarsen import rename_centered_xy_coords, shift_edge_var_to_center
from vcm.fv3_restarts import open_restarts, _parse_time_string, \
    _parse_first_last_forecast_times
from vcm.select import mask_to_surface_type

logger = logging.getLogger()
logger.setLevel(logging.INFO)

INIT_TIME_DIM = "initialization_time"
FORECAST_TIME_DIM = "forecast_time"
GRID_XY_COORDS =  {
    COORD_X_CENTER: range(48), COORD_Y_CENTER: range(48),
    COORD_X_OUTER: range(49), COORD_Y_OUTER: range(49)}
GRID_VARS = list(GRID_XY_COORDS.keys())+ ['area']
INPUT_VARS = ["sphum", "T", "delp", "u", "v", "slmsk"]
TARGET_VARS = ["Q1", "Q2", "QU", "QV"]


def run(args, pipeline_args):
    fs = gcsfs.GCSFileSystem(project=args.gcs_project)
    data_path = os.path.join(args.gcs_bucket, args.gcs_input_data_path)
    gcs_urls = ["gs://" + run_dir_path for run_dir_path in sorted(fs.ls(data_path))]
    _save_grid_spec(fs, gcs_urls[0], args.gcs_output_data_dir, args.gcs_bucket)
    data_batch_urls = _get_url_batches(gcs_urls, args.timesteps_per_output_file)

    print(f"Processing {len(data_batch_urls)} subsets...")
    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
                p
                | beam.Create(data_batch_urls)
                | "LoadCloudData" >> beam.Map(_load_cloud_data, fs=fs)
                | "CreateTrainingCols" >> beam.Map(_create_train_cols)
                | "MaskToSurfaceType"
                    >> beam.Map(
                        mask_to_surface_type,
                        surface_type=args.mask_to_surface_type)
                | "WriteToZarr"
                    >> beam.Map(
                        _write_to_zarr,
                        gcs_dest_dir=args.gcs_output_data_dir,
                        bucket=args.gcs_bucket)
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


def _write_to_zarr(
        ds, gcs_dest_dir, bucket="gs://vcm-ml-data", zarr_filename=None
):
    """Writes temporary zarr on worker and moves it to GCS

    Args:
        ds: xr dataset for single training batch
        gcs_dest_path: write location on GCS
        zarr_filename: name for zarr, use first timestamp as label
        bucket: GCS bucket
    Returns:
        None
    """
    logger.info("Writing to zarr...")
    if not zarr_filename:
        zarr_filename = _filename_from_first_timestep(ds)
    output_path = os.path.join(bucket, gcs_dest_dir, zarr_filename)
    ds.to_zarr(zarr_filename, mode="w")
    gsutil.copy(zarr_filename, output_path)
    logger.info(f"Done writing zarr to {output_path}")
    shutil.rmtree(zarr_filename)


def _filename_from_first_timestep(ds):
    timestep = min(ds[INIT_TIME_DIM].values).strftime("%Y%m%d.%H%M%S")
    return timestep + ".zarr"


def _load_cloud_data(run_dirs, fs):
    """

    Args:
        fs: GCSFileSystem
        run_dirs: list of GCS urls to open

    Returns:
        xarray dataset of concatenated zarrs in url list
    """
    logger.info(
        f"Using run dirs for batch: "
        f"{[os.path.basename(run_dir[:-1]) for run_dir in run_dirs]}")
    ds_runs = []
    for run_dir in run_dirs:
        t_init, t_last = _parse_first_last_forecast_times(fs, run_dir)
        ds_run = open_restarts(run_dir, t_init, t_last) \
            .rename({"time": FORECAST_TIME_DIM}) \
            .isel({FORECAST_TIME_DIM: slice(-2, None)}) \
            [INPUT_VARS] \
            .expand_dims(dim={INIT_TIME_DIM: [_parse_time_string(t_init)]}) \
            .assign_coords(GRID_XY_COORDS)
        ds_run = _set_forecast_time_coord(ds_run)
        ds_runs.append(ds_run)
    return xr.concat(ds_runs, INIT_TIME_DIM)


def _set_forecast_time_coord(ds):
    delta_t_forecast = (ds.forecast_time.values[1] - ds.forecast_time.values[
        0])
    ds.reset_index([FORECAST_TIME_DIM], drop=True)
    ds.assign_coords({FORECAST_TIME_DIM: [timedelta(seconds=0), delta_t_forecast]})
    return ds


def _save_grid_spec(fs, run_dir, gcs_output_data_dir, gcs_bucket):
    grid_info_files = ["gs://" + filename for filename in fs.ls(run_dir)
        if "atmos_dt_atmos" in filename]
    os.makedirs("temp_grid_spec", exist_ok=True)
    gsutil.copy_many(grid_info_files, "temp_grid_spec")
    grid = open_cubed_sphere(
        "temp_grid_spec/atmos_dt_atmos",
        num_subtiles=1,
        pattern='{prefix}.tile{tile:d}.nc'
    )[['area', VAR_LAT_OUTER, VAR_LON_OUTER, VAR_LAT_CENTER, VAR_LON_CENTER]]
    _write_to_zarr(
        grid, gcs_output_data_dir, gcs_bucket, zarr_filename="grid_spec.zarr")
    logger.info(f"Wrote grid spec to "
                f"{os.path.join(gcs_bucket, gcs_output_data_dir, 'grid_spec.zarr')}")
    shutil.rmtree("temp_grid_spec")


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
    ds = ds[cols_to_keep] \
            .isel({INIT_TIME_DIM: slice(None, ds.sizes[INIT_TIME_DIM] - 1),
                   FORECAST_TIME_DIM: 0}) \
            .squeeze(drop=True)
    return ds
