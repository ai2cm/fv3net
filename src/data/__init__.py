import xarray as xr

root = "/home/noahb/fv3net/"

def open_data(sources=False):
    ds = xr.open_zarr(root + "data/interim/2019-07-17-FV3_DYAMOND_0.25deg_15minute_256x256_blocks.zarr")

    if sources:
        src = xr.open_zarr(root + "data/interim/apparent_sources.zarr")
    return ds.merge(src)