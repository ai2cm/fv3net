import argparse
import gcsfs
import numpy as np
import os
import xarray as xr

from fv3net.machine_learning import reshape

from vcm.calc import apparent_source
#from vcm.calc.diag_ufuncs import mask_to_surface_type
from vcm.cubedsphere import shift_edge_var_to_center, rename_centered_xy_coords

GRID_VARS = [
    'grid_lon',
    'grid_lat',
    'grid_lont',
    'grid_latt'
]

VARS_TO_KEEP = [
    'sphum',
    'T',
    'delp',
    'u',
    'v',
    'slmsk'
]

TARGET_VARS = ['Q1', 'Q2', 'QU', 'QV']


def create_training_set(
        gcs_data_path,
        num_timesteps_to_sample,
        sample_dims=['tile', 'grid_yt', 'grid_xt', 'initialization_time'],
        chunk_size=5e5,
        bucket='vcm-ml-data',
        project='vcm-ml'
):
    ds = _load_cloud_data(
        gcs_data_path, num_timesteps_to_sample, bucket, project, t_dim)

    da_centered_u = rename_centered_xy_coords(shift_edge_var_to_center(ds['u']))
    da_centered_v = rename_centered_xy_coords(shift_edge_var_to_center(ds['v']))
    ds['QU'] = apparent_source(da_centered_u)
    ds['QV'] = apparent_source(da_centered_v)
    ds['Q1'] = apparent_source(ds.T)
    ds['Q2'] = apparent_source(ds.sphum)

    ds = ds[VARS_TO_KEEP + TARGET_VARS]
    ds = _reshape_and_shuffle(ds, sample_dims, chunk_size)
    return ds


def _reshape_and_shuffle(
        ds,
        sample_dims,
        chunk_size
):
    ds_stacked = ds.stack(sample=sample_dims).transpose("sample", "pfull")
    ds_chunked = ds_stacked.chunk({"sample": chunk_size})
    ds_chunked_shuffled = reshape.shuffled(ds_chunked, dim="sample")
    return ds_chunked_shuffled


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample-dims',
        type=str,
        nargs='+',
        help="coordinate dimensions to stack over"
    )
