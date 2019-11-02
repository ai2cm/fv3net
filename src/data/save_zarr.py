import xarray as xr
from os.path import join
from datetime import datetime, timedelta
import cftime
import pandas as pd
import gcsfs
from src.data.rundir import combine_file_categories



def write_cloud_zarr(ds, gcs_path):
    fs = gcsfs.GCSFileSystem(project='vcm-ml')
    mapping = fs.get_mapper(gcs_path)
    ds.to_zarr(store=mapping, mode='w')
    return ds


def open_cloud_zarr(gcs_path):
    fs = gcsfs.GCSFileSystem(project='vcm-ml')
    mapping = fs.get_mapper(gcs_path)
    return xr.open_zarr(fs)


def save_timestep_to_zarr(timestep, grid, bucket_dir):
    run_dir = f"./data/restart/{grid}/{timestep}/rundir"
    tile = pd.Index(range(1, N_TILES + 1), name='tile')
    time = pd.date_range(
        start=pd.to_datetime(timestep, format=TIME_FMT) + pd.Timedelta(15, unit='m'),
        end=pd.to_datetime(timestep, format=TIME_FMT), periods=2)
    new_dims = {'Time' : time, 'tile' : tile}
    ds = combine_file_categories(
        run_dir=run_dir,
        category_mapping=CATEGORY_DIR_MAPPING,
        new_dims=new_dims,
        grid_spec_mapping=GRID_SPEC_AXES_MAP,
        oro_data_mapping=ORO_DATA_AXES_MAP,
        tile_suffixes=TILE_SUFFIXES
    )
    write_cloud_zarr(ds, timestep, grid, bucket_dir)

