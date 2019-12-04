import argparse
import gcsfs
import logging
import os
import time
import xarray as xr

from fv3net.machine_learning import reshape

from vcm.calc import apparent_source
# from vcm.calc.diag_ufuncs import mask_to_surface_type
from vcm.cubedsphere import shift_edge_var_to_center, rename_centered_xy_coords

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
    'slmsk',
    'grid_latt',
    'grid_lont'
]

TARGET_VARS = ['Q1', 'Q2', 'QU', 'QV']


def create_training_set(
        gcs_data_path,
        num_timesteps_to_sample=None,
        sample_dims=['tile', 'grid_yt', 'grid_xt', 'initialization_time'],
        sample_chunk_size=5e5,
        bucket='vcm-ml-data',
        project='vcm-ml'
):
    ds = _load_cloud_data(
        gcs_data_path, num_timesteps_to_sample, bucket, project)
    da_centered_u = rename_centered_xy_coords(shift_edge_var_to_center(ds['u']))
    da_centered_v = rename_centered_xy_coords(shift_edge_var_to_center(ds['v']))
    ds['u'] = da_centered_u
    ds['v'] = da_centered_v
    ds['QU'] = apparent_source(ds.u)
    ds['QV'] = apparent_source(ds.v)
    ds['Q1'] = apparent_source(ds.T)
    ds['Q2'] = apparent_source(ds.sphum)

    ds = ds[VARS_TO_KEEP + TARGET_VARS] \
        .isel(forecast_time=0).squeeze().drop('forecast_time')
    if num_timesteps_to_sample:
        # when using a subset of timesteps, the consecutive steps are loaded to compute
        # tendencies but must be discarded since their calculate tendencies have much
        # larger dt than the simulation step
        ds = ds.isel(initialization_time=[2 * i for i in range(num_timesteps_to_sample)])
    ds = _reshape_and_shuffle(ds, sample_dims, sample_chunk_size)
    return ds


def write_to_zarr(
        ds,
        gcs_dest_path,
        load_size,
        bucket='vcm-ml-data',
        project='vcm-ml'
):
    """Still haven't figured out why writing is so slow

    Args:
        ds:
        gcs_dest_path:
        load_size:

    Returns:

    """
    num_samples = len(ds.sample)
    num_chunks = int(num_samples / load_size) + 1
    logger.info("Writing to zarr...")
    fs = gcsfs.GCSFileSystem(project=project)
    output_path = fs.get_mapper(os.path.join(bucket, gcs_dest_path))
    for i in range(num_chunks):
        t0 = time.time()
        load_slice = slice(i * load_size, (i + 1) * load_size)
        ds_subset = ds.isel(sample=load_slice).load()
        ds_subset.to_zarr(output_path, 'a', append_dim='sample')
        del ds_subset
        logger.info(f"{i}/{num_chunks} written, {int(time.time() - t0)} s.")
    logger.info("Done writing zarr.")


def _reshape_and_shuffle(
        ds,
        sample_dims,
        chunk_size
):
    ds_stacked = ds.stack(sample=sample_dims).transpose("sample", "pfull")
    ds_chunked = ds_stacked.chunk({"sample": chunk_size})
    ds_chunked_shuffled = reshape.shuffled(ds_chunked, dim="sample") \
        .reset_index('sample')
    return ds_chunked_shuffled


def _load_cloud_data(
        gcs_data_path,
        num_timesteps_to_sample,
        bucket,
        project
):
    fs = gcsfs.GCSFileSystem(project=project)
    zarr_path = os.path.join(bucket, gcs_data_path)
    zarr_urls = sorted(fs.ls(zarr_path))
    logger.info(f"{len(zarr_urls)} zarrs in {zarr_path}...")
    if num_timesteps_to_sample:
        sample_urls = _select_samples(zarr_urls, num_timesteps_to_sample)
    else:
        sample_urls = zarr_urls
    logger.info(f"Sampled {len(sample_urls)} zarrs...")
    gcs_zarr_mappings = [fs.get_mapper(url) for url in sample_urls]
    logger.info(f"Loading zarrs from {zarr_path}...")
    t0 = time.time()
    ds = xr.combine_by_coords(
        map(xr.open_zarr, gcs_zarr_mappings),
        data_vars=VARS_TO_KEEP
    )
    logger.info(f"Finished loading zarrs. {int(time.time() - t0)} s")
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
        zarr_urls[round(len(zarr_urls) / num_timesteps_to_sample) * i]
        for i in range(num_timesteps_to_sample)]
    # cannot sample last timestep as we need the next timestep for tendencies
    if initial_time_urls[-1] == zarr_urls[-1]:
        initial_time_urls = initial_time_urls[:-1]
    next_time_urls = [
        zarr_urls[zarr_urls.index(file_to_sample) + 1]
        for file_to_sample in initial_time_urls]

    sample_urls = sorted(initial_time_urls + next_time_urls)
    return sample_urls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gcs-input-data-path',
        type=str,
        required=True,
        help="Location of input data in Google Cloud Storage bucket. "
             "Don't include bucket in path."
    )
    parser.add_argument(
        '--gcs-output-data-path',
        type=str,
        required=True,
        help="Write path for train data in Google Cloud Storage bucket. "
             "Don't include bucket in path."
    )
    parser.add_argument(
        '--gcs-bucket',
        type=str,
        default='vcm-ml-data',
        help="Google Cloud Storage bucket name."
    )
    parser.add_argument(
        '--gcs-project',
        type=str,
        default='vcm-ml',
        help="Project name for google cloud."
    )
    parser.add_argument(
        '--num-timesteps-to-sample',
        type=int,
        default=None,
        help="If not using full dataset, specify number of timesteps to sample."
    )
    parser.add_argument(
        '--sample-dims',
        type=str,
        nargs='+',
        default=['tile', 'grid_yt', 'grid_xt', 'initialization_time'],
        help="dimensions to stack into sample index"
    )
    parser.add_argument(
        '--sample-chunk-size',
        type=int,
        default=5e5,
        help="Chunk size in the sample dimension after time/space dims are stacked."
    )
    args = parser.parse_args()
    ds = create_training_set(
        gcs_data_path=args.gcs_input_data_path,
        num_timesteps_to_sample=args.num_timesteps_to_sample,
        sample_dims=args.sample_dims,
        sample_chunk_size=args.sample_chunk_size,
        bucket=args.gcs_bucket,
        project=args.gcs_project
    )
    write_to_zarr(
        ds,
        gcs_dest_path=args.gcs_output_data_path,
        load_size=args.sample_chunk_size,
        bucket=args.gcs_bucket,
        project=args.gcs_project
    )