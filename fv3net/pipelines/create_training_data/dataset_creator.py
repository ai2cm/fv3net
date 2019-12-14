import argparse
import gcsfs
import logging
import os
import time
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
    """Still haven't figured out why writing is so slow

    Args:
        ds:
        gcs_dest_path:
        load_size:

    Returns:

    """
    logger.info("Writing to zarr...")
    t0 = time.time()
    output_path = os.path.join(bucket, gcs_dest_dir, zarr_filename)
    ds.to_zarr(zarr_filename, mode="w")
    gsutil.copy(zarr_filename, output_path)
    logger.info(f"Done writing zarr to {output_path}, {int(time.time() - t0)} s.")


def create_training_dataset(
        data_urls,
        mask_to_surface_type=None,
        project='vcm-ml'
):
    t0 = time.time()
    fs = gcsfs.GCSFileSystem(project=project)
    ds = _load_cloud_data(fs, data_urls)
    logger.info(f"Finished loading zarrs for timesteps "
                f"{[get_timestep_from_filename(url) for url in data_urls]}. "
                f"{int(time.time() - t0)} s")
    ds = _create_train_cols(ds)
    if not mask_to_surface_type:
        ds = mask_to_surface_type(ds, mask_to_surface_type)
    return ds


def _load_cloud_data(fs, gcs_urls):
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
    if surface_type not in ['sea', 'land', 'seaice']:
        raise ValueError("Must mask to surface_type in ['sea', 'land', 'seaice'].")
    surface_type_codes = {'sea': 0, 'land': 1, 'seaice': 2}
    mask = ds.slmsk == surface_type_codes[surface_type]
    ds_masked = ds.where(mask)
    return ds_masked


def _create_train_cols(ds):
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







