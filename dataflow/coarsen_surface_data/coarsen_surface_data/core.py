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
    return cubedsphere.block_median(x.chunk(), coarsening).compute()


@delayed
def upload_to_gcs(src, dest, save_op):
    logging.info("uploading %s to %s" % (src, dest))
    subprocess.check_call(['gsutil', '-q', 'cp',  src, dest])
    logging.info("uploading %s done" % dest)


@delayed
def run_all_delayeds(*args):
    pass


def output_names(key) -> dict:
    timestep, coarsenings = key
    urls = {}
    for coarsening in coarsenings:
        res = int(3072 // coarsening)
        category = 'sfc_data'
        output_file_name = f'gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/coarsened/C{res}/{timestep}/{category}.nc'
        urls[coarsening] = output_file_name
    return urls


def coarsen_and_upload_surface(key):
    timestep, coarsenings= key
    output_file_names = output_names(key)

    ops = []

    # data downloading ops
    category = 'sfc_data'
    stored_resolution = 3702
    files = utils.file_names_for_time_step(timestep, category, resolution=stored_resolution)
    grouped_files = utils.group_file_names(files)
    opened = utils.map_ops(utils._open_remote_nc, grouped_files) 

    # coarse-graining
    with tempfile.TemporaryDirectory() as d:
        for coarsening in coarsenings:
            output_file_name = output_file_names[coarsening]
            logging.info("beggining processing job to %s" % output_file_name)
            coarse = utils.map_ops(_median_no_dask, opened, coarsening) 
            tiles = {key: combine_subtiles(val)
                    for key, val in coarse.items()}
            ds = concat_files(tiles)
            name = f"{d}/{coarsening}.name"
            save_op = ds.to_netcdf(name)
            upload_op = upload_to_gcs(name, output_file_name, save_op)
            ops.append(upload_op)

        all_ops = run_all_delayeds(*ops)
        #all_ops.compute(scheduler="single-threaded")
        all_ops.compute()

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    coarsen_and_upload_surface(timestep='20160805.114500', coarsenings=(8, 16, 32, 64))
