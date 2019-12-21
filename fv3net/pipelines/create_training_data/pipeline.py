import numba
import argparse
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import gcsfs
import logging
import os
import xarray as xr

from vcm.calc import apparent_source
from vcm.cloud import gsutil
from vcm.cubedsphere.coarsen import rename_centered_xy_coords, shift_edge_var_to_center
from vcm.select import mask_to_surface_type

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# There have been issues where python crashes immediately unless numba gets
# imported explicitly before vcm, hence the import and del
del numba


TIME_DIM = "initialization_time"
GRID_VARS = ["grid_lon", "grid_lat", "grid_lont", "grid_latt"]
INPUT_VARS = ["sphum", "T", "delp", "u", "v", "slmsk"]
TARGET_VARS = ["Q1", "Q2", "QU", "QV"]


def run(args, pipeline_args):
    fs = gcsfs.GCSFileSystem(project=args.gcs_project)

    zarr_dir = os.path.join(args.gcs_bucket, args.gcs_input_data_path)
    gcs_urls = sorted(fs.ls(zarr_dir))
    num_outputs = int((len(gcs_urls) - 1) / args.timesteps_per_output_file)
    tstep_pairs = [
        (
            args.timesteps_per_output_file * i,
            args.timesteps_per_output_file * i + (args.timesteps_per_output_file + 1),
        )
        for i in range(num_outputs)
    ]
    data_urls = [gcs_urls[start_ind:stop_ind] for start_ind, stop_ind in tstep_pairs]

    print(f"Processing {len(data_urls)} subsets...")
    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | beam.Create(data_urls)
            | "LoadCloudData" >> beam.Map(_load_cloud_data, fs=fs)
            | "CreateTrainingCols" >> beam.Map(_create_train_cols)
            | "MaskToSurfaceType"
            >> beam.Map(mask_to_surface_type, surface_type=args.mask_to_surface_type)
            | "WriteToZarr"
            >> beam.Map(
                _write_to_zarr,
                gcs_dest_dir=args.gcs_output_data_dir,
                bucket=args.gcs_bucket,
            )
        )


def _write_to_zarr(
    ds, gcs_dest_dir, bucket="gs://vcm-ml-data",
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
    zarr_filename = _filename_from_first_timestep(ds)
    output_path = os.path.join(bucket, gcs_dest_dir, zarr_filename)
    ds.to_zarr(zarr_filename, mode="w")
    gsutil.copy(zarr_filename, output_path)
    logger.info(f"Done writing zarr to {output_path}")


def _filename_from_first_timestep(ds):
    timestep = min(ds[TIME_DIM].values).strftime("%Y%m%d.%H%M%S")
    return timestep + ".zarr"


def _load_cloud_data(gcs_urls, fs):
    """

    Args:
        fs: GCSFileSystem
        gcs_urls: list of GCS urls to open

    Returns:
        xarray dataset of concatenated zarrs in url list
    """
    gcs_zarr_mappings = [fs.get_mapper(url) for url in gcs_urls]
    ds = xr.concat(map(xr.open_zarr, gcs_zarr_mappings), TIME_DIM)[
        INPUT_VARS + GRID_VARS
    ]
    return ds


def _create_train_cols(ds, cols_to_keep=INPUT_VARS + TARGET_VARS + GRID_VARS):
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
    num_slices = len(ds.initialization_time.values) - 1
    ds = ds[cols_to_keep].isel({TIME_DIM: slice(None, num_slices)})
    if "forecast_time" in ds.dims:
        ds = ds.isel(forecast_time=0).squeeze().drop("forecast_time")
    return ds
