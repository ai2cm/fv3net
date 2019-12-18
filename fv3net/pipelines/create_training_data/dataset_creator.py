import gcsfs
import logging
import os
import xarray as xr

from vcm.calc import apparent_source
from vcm.cloud import gsutil
from vcm.convenience import get_timestep_from_filename
from vcm.cubedsphere import shift_edge_var_to_center, rename_centered_xy_coords

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler('dataset.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)


GRID_VARS = [
    'grid_lon',
    'grid_lat',
    'grid_lont',
    'grid_latt'
]

INPUT_VARS = [
    'sphum',
    'T',
    'delp',
    'u',
    'v',
    'slmsk'
]

TARGET_VARS = ['Q1', 'Q2', 'QU', 'QV']


def write_to_zarr(
        ds,
        gcs_dest_dir,
        zarr_filename,
        bucket='vcm-ml-data',
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
    output_path = os.path.join(bucket, gcs_dest_dir, zarr_filename)
    ds.to_zarr(zarr_filename, mode="w")
    gsutil.copy(zarr_filename, output_path)
    logger.info(f"Done writing zarr to {output_path}")


def create_training_dataset(
        data_urls,
        mask_to_surface_type=None,
        project='vcm-ml'
):
    """

    Args:
        data_urls:
        mask_to_surface_type:
        project:

    Returns:

    """
    fs = gcsfs.GCSFileSystem(project=project)
    ds = _load_cloud_data(fs, data_urls)
    logger.info(f"Finished loading zarrs for timesteps "
                f"{[get_timestep_from_filename(url) for url in data_urls]}. ")
    ds = _create_train_cols(ds)
    if not mask_to_surface_type:
        ds = mask_to_surface_type(ds, mask_to_surface_type)
    return ds


def _load_cloud_data(fs, gcs_urls):
    """

    Args:
        fs: GCSFileSystem
        gcs_urls: list of GCS urls to open

    Returns:
        xarray dataset of concatenated zarrs in url list
    """
    gcs_zarr_mappings = [fs.get_mapper(url) for url in gcs_urls]
    ds = xr.concat(
        map(xr.open_zarr, gcs_zarr_mappings),
        'initialization_time'
    )[INPUT_VARS + GRID_VARS]
    return ds


def mask_to_surface_type(
        ds,
        surface_type
):
    """

    Args:
        ds: xarray dataset, must have variable slmsk
        surface_type: one of ['sea', 'land', 'seaice']

    Returns:
        input dataset masked to the surface_type specified
    """
    if surface_type not in ['sea', 'land', 'seaice']:
        raise ValueError("Must mask to surface_type in ['sea', 'land', 'seaice'].")
    surface_type_codes = {'sea': 0, 'land': 1, 'seaice': 2}
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
    da_centered_u = rename_centered_xy_coords(shift_edge_var_to_center(ds['u']))
    da_centered_v = rename_centered_xy_coords(shift_edge_var_to_center(ds['v']))
    ds['u'] = da_centered_u
    ds['v'] = da_centered_v
    ds['QU'] = apparent_source(ds.u)
    ds['QV'] = apparent_source(ds.v)
    ds['Q1'] = apparent_source(ds.T)
    ds['Q2'] = apparent_source(ds.sphum)
    num_slices = len(ds.initialization_time.values) - 1
    ds = ds[INPUT_VARS + TARGET_VARS + GRID_VARS] \
        .isel(forecast_time=0).squeeze().drop('forecast_time') \
        .isel(initialization_time=slice(None, num_slices))
    return ds







