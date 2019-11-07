import xarray as xr
import intake
import yaml
from src import TOP_LEVEL_DIR
from .remote_data import open_gfdl_data_with_2d, open_gfdl_15_minute_SHiELD
from pathlib import Path

# TODO Fix short tag yaml file get for fv3 installed as package


def get_root():
    """Returns the absolute path to the root directory for any machine"""
    return str(TOP_LEVEL_DIR)


def get_shortened_dataset_tags():
    """"Return a dictionary mapping short dataset definitions to the full names"""
    short_dset_yaml = Path(TOP_LEVEL_DIR) / "short_datatag_defs.yml"
    return yaml.load(short_dset_yaml.open(), Loader=yaml.SafeLoader)


root = get_root()


# TODO: I believe fv3_data_root and paths are legacy
fv3_data_root = "/home/noahb/data/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute/"

paths = {
    "1deg_src": f"{root}/data/interim/advection/2019-07-17-FV3_DYAMOND_0.25deg_15minute_regrid_1degree.zarr"
}


def open_catalog():
    return intake.Catalog(f"{root}/catalog.yml")


def open_dataset(tag) -> xr.Dataset:
    short_dset_tags = get_shortened_dataset_tags()
    if tag == "gfdl":
        return open_gfdl_data_with_2d(open_catalog())
    elif tag == "1degTrain":
        return (
            open_dataset("1deg")
            .merge(xr.open_zarr(paths["1deg_src"]))
            .pipe(_insert_apparent_heating)
        )
    else:
        if tag in short_dset_tags:
            # TODO: Debug logging about tag transformation
            tag = short_dset_tags[tag]

        curr_catalog = open_catalog()
        source = curr_catalog[tag]
        dataset = source.to_dask()

        for transform in source.metadata.get("data_transforms", ()):
            print(f"Applying data transform: {transform}")
            transform_function = globals()[transform]
            dataset = transform_function(dataset)

        return dataset


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


## Data Adjustments ##
def _rename_SHiELD_varnames_to_orig(ds: xr.Dataset) -> xr.Dataset:
    """
    Replace varnames from new dataset to match original style 
    from initial DYAMOND data.
    """

    rename_list = {
        "ucomp": "u",
        "vcomp": "v",
        "sphum": "qv",
        "HGTsfc": "zs",
        "delz": "dz",
        "delp": "dp",
    }
    return ds.rename(rename_list)


def replace_esmf_coords_reg_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    Replace ESMF coordinates from regridding to a regular lat/lon grid
    """

    lat_1d = ds.lat.isel(x=0).values
    lon_1d = ds.lon.isel(y=0).values
    ds = ds.assign_coords({"x": lon_1d, "y": lat_1d})
    ds = ds.rename({"lat": "lat_grid", "lon": "lon_grid"})

    return ds


def _replace_esmf_coords(ds):
    #  Older version using DYAMOND data -AP Oct 2019
    return (
        # DYAMOND data
        ds.assign_coords(x=ds.lon.isel(y=0), y=ds.lat.isel(x=0))
        .drop(["lat", "lon"])
        .rename({"x": "lon", "y": "lat"})
    )


def _insert_apparent_heating(ds: xr.Dataset) -> xr.Dataset:
    from .calc import apparent_heating

    dqt_dt = ds.advection_qv + ds.storage_qv
    dtemp_dt = ds.advection_temp + ds.storage_temp
    return ds.assign(q1=apparent_heating(dtemp_dt, ds.w), q2=dqt_dt)
