import gcsfs
import numpy as np
import os
import xarray as xr

from vcm.calc import apparent_source
from vcm.calc.diag_ufuncs import mask_to_surface_type


GRID_VARS = [
    'grid_lon',
    'grid_lat',
    'grid_lont',
    'grid_latt'
]

VARS_TO_KEEP = [
    'sphum',
    'T',
    'area',
    'delp',
    'u',
    'v',
    'slmsk'
]


def create_training_set(
        gcs_data_path,
        num_timesteps_to_sample,
        bucket,
        project,
        t_dim
):
    pass


def _load_cloud_data(
        gcs_data_path,
        num_timesteps_to_sample,
        bucket,
        project,
        t_dim
):
    fs = gcsfs.GCSFileSystem(project=project)
    zarr_path = os.path.join(bucket, gcs_data_path)
    zarr_urls = sorted(fs.ls(zarr_path))
    sample_urls = _select_samples(zarr_urls, num_timesteps_to_sample)
    gcs_zarr_mappings = [fs.get_mapper(url) for url in sample_urls]
    ds = xr.combine_by_coords(
        map(xr.open_zarr, gcs_zarr_mappings),
        data_vars=VARS_TO_KEEP
    ) \
        .chunk({t_dim: 2})
    return ds


def _select_samples(
        zarr_urls,
        num_timesteps_to_sample,

):
    """Sample evenly in hour/min intervals so that there is even coverage of diurnal
    cycle in training data. Returns the urls to read data from, plus the urls for the
    next consecutive timestep (used to calculate tendencies).

    Args:
        time_ordered_urls:
        num_tsteps_to_sample:

    Returns:

    """
    initial_time_urls = [
        zarr_urls[round(len(zarr_urls) / num_timesteps_to_sample)*i]
            for i in range(num_timesteps_to_sample)]
    # cannot sample last timestep as we need the next timestep for tendencies
    if initial_time_urls[-1] == zarr_urls[-1]:
        initial_time_urls = initial_time_urls[:-1]
    next_time_urls = [
        zarr_urls[zarr_urls.index(file_to_sample)+1]
            for file_to_sample in initial_time_urls]

    sample_urls = sorted(initial_time_urls + next_time_urls)
    return sample_urls


