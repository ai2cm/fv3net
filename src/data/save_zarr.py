import xarray as xr
from os.path import join
import pandas as pd
import gcsfs

data_id = "data/raw/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute"
output_2d = "data/interim/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute_2d.zarr"
output_3d = "data/interim/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute_3d.zarr"

files_3d = [
    "h_plev_C3072_1536x768.fre.nc",
    "qi_plev_C3072_1536x768.fre.nc",
    "ql_plev_C3072_1536x768.fre.nc",
    "q_plev_C3072_1536x768.fre.nc",
    "t_plev_C3072_1536x768.fre.nc",
    "u_plev_C3072_1536x768.fre.nc",
    "v_plev_C3072_1536x768.fre.nc",
    "pres_C3072_1536x768.fre.nc",
    "qi_C3072_1536x768.fre.nc",
    "ql_C3072_1536x768.fre.nc",
    "qr_C3072_1536x768.fre.nc",
    "qv_C3072_1536x768.fre.nc",
    "temp_C3072_1536x768.fre.nc",
    "w_C3072_1536x768.fre.nc",
    "u_C3072_1536x768.fre.nc",
    "v_C3072_1536x768.fre.nc",
]

files_2d = [
    "cape_C3072_1536x768.fre.nc",
    "cin_C3072_1536x768.fre.nc",
    "cldc_C3072_1536x768.fre.nc",
    "flds_C3072_1536x768.fre.nc",
    "flus_C3072_1536x768.fre.nc",
    "flut_C3072_1536x768.fre.nc",
    "fsds_C3072_1536x768.fre.nc",
    "fsdt_C3072_1536x768.fre.nc",
    "fsus_C3072_1536x768.fre.nc",
    "fsut_C3072_1536x768.fre.nc",
    "h500_C3072_1536x768.fre.nc",
    "intqg_C3072_1536x768.fre.nc",
    "intqi_C3072_1536x768.fre.nc",
    "intql_C3072_1536x768.fre.nc",
    "intqr_C3072_1536x768.fre.nc",
    "intqs_C3072_1536x768.fre.nc",
    "intqv_C3072_1536x768.fre.nc",
    "lhflx_C3072_1536x768.fre.nc",
    "pr_C3072_1536x768.fre.nc",
    "ps_C3072_1536x768.fre.nc",
    "q2m_C3072_1536x768.fre.nc",
    "rh500_C3072_1536x768.fre.nc",
    "rh700_C3072_1536x768.fre.nc",
    "rh850_C3072_1536x768.fre.nc",
    "shflx_C3072_1536x768.fre.nc",
    "t2m_C3072_1536x768.fre.nc",
    "ts_C3072_1536x768.fre.nc",
    "u10m_C3072_1536x768.fre.nc",
    "u200_C3072_1536x768.fre.nc",
    "ustrs_C3072_1536x768.fre.nc",
    "v10m_C3072_1536x768.fre.nc",
    "qs_C3072_1536x768.fre.nc",
    "v200_C3072_1536x768.fre.nc",
    "vstrs_C3072_1536x768.fre.nc",
]

BOTH_DIRS = ['RESTART', 'INPUT']
CATEGORY_DIR_MAPPING = {
    'grid_spec' : ['.'],
    'oro_data' : ['INPUT'],
    'fv_core.res' : BOTH_DIRS,
    'fv_srf_wnd.res' : BOTH_DIRS,
    'fv_tracer.res' : BOTH_DIRS,
    'sfc_data' : BOTH_DIRS,
    'phy_data' : ['RESTART']
}

GRID_SPEC_AXES_MAP = {
    'grid_x' : 'xaxis_2',
    'grid_y' : 'yaxis_1',
    'grid_xt' : 'xaxis_1',
    'grid_yt' : 'yaxis_2'
}

ORO_DATA_AXES_MAP = {
    'lon' : 'xaxis_1',
    'lat' : 'yaxis_2'
}

N_TILES = 6
TILE_SUFFIXES = [f".tile{tile}.nc" for tile in range(1, N_TILES + 1)]
TIME_FMT='%Y%m%d.%H%M%S'


def open_files(files, **kwargs):
    paths = [join(data_id, path) for path in files]
    return xr.open_mfdataset(paths, **kwargs)


def cubed_sphere_tile_paths(
    run_dir,
    category,
    tile_suffixes,
    target_dirs = ['RESTART', 'INPUT']
):
    return [
        [join(join(run_dir, target_dir), category + tile_suffix) for tile_suffix in tile_suffixes]
        for target_dir in target_dirs]


def assign_time_dim(ds, dirs, time_coords):
    if len(dirs) > 1:
        ds = ds.assign_coords(Time=time_coords)
    elif dirs == ['RESTART']:
        ds = ds.assign_coords(Time=[time_coords[0]])
    elif dirs == ['INPUT']:
        ds = ds.assign_coords(Time=[time_coords[1]])
    return ds


def open_oro_data(paths, oro_data_mapping):
    ds = xr.concat(
        objs=[xr.open_dataset(path).drop(labels=['lat', 'lon']) for path in paths[0]],
        dim='tile'
    )
    ds = ds.rename(oro_data_mapping)
    return ds.drop(labels='slmsk')


def add_vertical_coords(ds, run_dir, time_coords):
    vertical_coords_ds = xr.open_dataset(join(run_dir, 'INPUT/fv_core.res.nc')).rename({'xaxis_1' : 'zaxis_2'})
    vertical_coords_ds = vertical_coords_ds.assign_coords(Time=[time_coords[1]])
    return xr.merge([ds, vertical_coords_ds])


def combine_file_categories(
    run_dir,
    category_mapping,
    tile_suffixes,
    new_dims={},
    grid_spec_mapping={},
    oro_data_mapping={}
):
    ds_dict = {}
    for (category, dirs) in category_mapping.items():
        paths = cubed_sphere_tile_paths(
            run_dir=run_dir,
            category=category,
            tile_suffixes=tile_suffixes,
            target_dirs=dirs
        )
        if category != 'oro_data':
            ds = xr.open_mfdataset(
                paths=paths,
                concat_dim=['Time', 'tile'],
                combine='nested'
            )
        else:
            ds = open_oro_data(paths, oro_data_mapping)
        if new_dims and 'tile' in new_dims.keys():
            ds = ds.assign_coords(tile=new_dims['tile'])
        if category == 'grid_spec' and grid_spec_mapping:
            ds = ds.squeeze(dim='Time').drop(labels='time').rename(grid_spec_mapping)
        ds = assign_time_dim(ds, dirs, new_dims['Time'])
        ds_dict[category] = ds
    ds_merged = xr.merge([ds for ds in ds_dict.values()])
    return add_vertical_coords(ds=ds_merged, run_dir=run_dir, time_coords=new_dims['Time'])


def write_cloud_zarr(ds, timestep, grid, bucket_dir):
    fs = gcsfs.GCSFileSystem(project='vcm-ml')
    path = f'/vcm-ml-data/{bucket_dir}/{grid}/restarted_at_{timestep}.zarr'
    print(path)
    mapping = fs.get_mapper(path)
    ds.to_zarr(store=mapping, mode='w')


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
    print(ds)
    write_cloud_zarr(ds, timestep, grid, bucket_dir)


def main():
    ds_3d = open_files(files_3d)
    ds_3d.to_zarr(output_3d, mode="w")

    ds_2d = open_files(files_2d)
    ds_2d.to_zarr(output_2d, mode="w")


if __name__ == "__main__":
    main()
