import xarray as xr
import intake
from .remote_data import open_gfdl_data_with_2d, open_gfdl_15_minute_data_with_2d
from pathlib import Path


def get_root():
    """Returns the absolute path to the root directory for any machine"""
    return str(Path(__file__).absolute().parent.parent.parent)


root = get_root()
fv3_data_root = (
    "/home/noahb/data/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute/"
)

paths = {
    "1deg": f"{root}/data/interim/2019-07-17-FV3_DYAMOND_0.25deg_15minute_regrid_1degree.zarr/",
    "1deg_src": f"{root}/data/interim/advection/2019-07-17-FV3_DYAMOND_0.25deg_15minute_regrid_1degree.zarr",
}


def open_catalog():
    return intake.Catalog(f"{root}/catalog.yml")


def open_dataset(tag) -> xr.Dataset:
    # TODO: add dataset name argument for these cases
    if tag == "gfdl":
        return open_gfdl_data_with_2d(open_catalog())
    elif tag == "gfdl_15_minute":
        return open_gfdl_15_minute_data_with_2d(open_catalog(), dataset_name = '2019-09-10-GFDL-SHiELD-15-minute-2-days')
    elif tag == "1deg":
        return xr.open_zarr(paths["1deg"]).pipe(_replace_esmf_coords)
    elif tag == "1degTrain":
        return (
            open_dataset("1deg")
            .merge(xr.open_zarr(paths["1deg_src"]))
            .pipe(_insert_apparent_heating)
        )
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


def _replace_esmf_coords(ds):
    return (
        ds.assign_coords(x=ds.lon.isel(y=0), y=ds.lat.isel(x=0))
        .drop(["lat", "lon"])
        .rename({"x": "lon", "y": "lat"})
    )


def _insert_apparent_heating(ds: xr.Dataset) -> xr.Dataset:
    from .calc import apparent_heating

    dqt_dt = ds.advection_qv + ds.storage_qv
    dtemp_dt = ds.advection_temp + ds.storage_temp
    return ds.assign(q1=apparent_heating(dtemp_dt, ds.w), q2=dqt_dt)
