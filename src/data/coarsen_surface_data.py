from src import utils
import pandas as pd
import xarray as xr
from dask.delayed import delayed
from src.data import cubedsphere
import subprocess
import logging
import tempfile

combine_subtiles = delayed(cubedsphere.combine_subtiles)


@delayed
def concat_files(tiles):
    tile_nums = [int(tile) for tile in tiles]
    idx = pd.Index(tile_nums, name='tile')
    return xr.concat(tiles.values(), dim=idx).sortby('tile')


@delayed
def _median_no_dask(x, coarsening):
    n = len(x['xaxis_1'])
    target_res = n // coarsening
    return cubedsphere.block_median(x.chunk(), target_res).compute()


@delayed
def upload_to_gcs(src, dest, save_op):
    logging.info("uploading %s to %s" % (src, dest))
    subprocess.check_call(['gsutil', '-q', 'cp',  src, dest])


def coarsen_and_upload_surface(timestep):
    category = 'sfc_data'
    stored_resolution = 3702
    coarsening = 8
    output_file_name = f'gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/coarsened/C384/{timestep}/{category}.nc'
    logging.info("Saving file to %s"%output_file_name)
    
    files = utils.file_names_for_time_step(timestep, category, resolution=stored_resolution)
    grouped_files = utils.group_file_names(files)
    opened = utils.map_ops(utils._open_remote_nc, grouped_files) 
    coarse = utils.map_ops(_median_no_dask, opened, coarsening) 
    tiles = {key: combine_subtiles(val)
            for key, val in coarse.items()}
    ds = concat_files(tiles)

    with tempfile.NamedTemporaryFile() as fp:
        save_op = ds.to_netcdf(fp.name)
        upload_op = upload_to_gcs(fp.name, output_file_name, save_op)
        upload_op.compute(scheduler="single-threaded")
        logging.info("uploading %s done" % output_file_name)
