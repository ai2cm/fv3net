import xarray as xr
import intake
from .remote_data import open_gfdl_data

root = "/home/noahb/fv3net/"
fv3_data_root = "/home/noahb/data/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute/"


def open_catalog():
    return intake.Catalog(f"{root}/catalog.yml")


def open_dataset(tag) -> xr.Dataset:
    if tag == "gfdl":
        return open_gfdl_data(open_catalog())
    else:
        raise NotImplementedError


def open_data(sources=False, two_dimensional=True):
    ds = xr.open_zarr(
        root
        + "data/interim/2019-07-17-FV3_DYAMOND_0.25deg_15minute_256x256_blocks.zarr"
    )

    if sources:
        src = xr.open_zarr(root + "data/interim/apparent_sources.zarr")
        ds = ds.merge(src)

    if two_dimensional:
        data_2d = xr.open_zarr(fv3_data_root + "2d.zarr")
        ds = xr.merge([ds, data_2d], join="left")

    return ds
