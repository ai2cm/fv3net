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
from vcm.cubedsphere import rename_centered_xy_coords, shift_edge_var_to_center

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("dataset.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


# There have been issues where python crashes immediately unless numba gets
# imported explicitly before vcm, hence the import and del
del numba


TIME_DIM = 'initialization_time'
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
            | "MaskToSurfaceType" >> beam.Map(
                _mask_to_surface_type, surface_type=args.mask_to_surface_type)
            | "WriteToZarr" >> beam.Map(
                _write_to_zarr,
                gcs_dest_dir=args.gcs_output_data_dir)
        )


def _write_to_zarr(
    ds, gcs_dest_dir, bucket="vcm-ml-data",
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


def _load_cloud_data(fs, gcs_urls):
    """

    Args:
        fs: GCSFileSystem
        gcs_urls: list of GCS urls to open

    Returns:
        xarray dataset of concatenated zarrs in url list
    """
    gcs_zarr_mappings = [fs.get_mapper(url) for url in gcs_urls]
    ds = xr.concat(map(xr.open_zarr, gcs_zarr_mappings), "initialization_time")[
        INPUT_VARS + GRID_VARS
    ]
    return ds


def _mask_to_surface_type(ds, surface_type):
    """

    Args:
        ds: xarray dataset, must have variable slmsk
        surface_type: one of ['sea', 'land', 'seaice']

    Returns:
        input dataset masked to the surface_type specified
    """
    if not surface_type:
        return ds
    elif surface_type not in ["sea", "land", "seaice"]:
        raise ValueError("Must mask to surface_type in ['sea', 'land', 'seaice'].")
    surface_type_codes = {"sea": 0, "land": 1, "seaice": 2}
    mask = ds.slmsk == surface_type_codes[surface_type]
    ds_masked = ds.where(mask)
    return ds_masked


def _create_train_cols(ds):
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
    ds = (
        ds[INPUT_VARS + TARGET_VARS + GRID_VARS]
        .isel(forecast_time=0)
        .squeeze()
        .drop("forecast_time")
        .isel(initialization_time=slice(None, num_slices))
    )
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gcs-input-data-path",
        type=str,
        required=True,
        help="Location of input data in Google Cloud Storage bucket. "
        "Don't include bucket in path.",
    )
    parser.add_argument(
        "--gcs-output-data-dir",
        type=str,
        required=True,
        help="Write path for train data in Google Cloud Storage bucket. "
        "Don't include bucket in path.",
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default="gs://vcm-ml-data",
        help="Google Cloud Storage bucket name.",
    )
    parser.add_argument(
        "--gcs-project",
        type=str,
        default="vcm-ml",
        help="Project name for google cloud.",
    )
    parser.add_argument(
        "--mask-to-surface-type",
        type=str,
        default=None,
        help="Mask to surface type in ['sea', 'land', 'seaice'].",
    )
    parser.add_argument(
        "--timesteps-per-output-file",
        type=int,
        default=2,
        help="Number of consecutive timesteps to calculate features/targets for in "
        "a single process and save to output file."
        "When the full output is shuffled at the data generator step, these"
        "timesteps will always be in the same training data batch.",
    )
    args, pipeline_args = parser.parse_known_args()

    """Main function"""
    run(args=args, pipeline_args=pipeline_args)
