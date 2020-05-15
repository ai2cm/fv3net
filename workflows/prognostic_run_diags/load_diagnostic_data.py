import intake
import datetime
import xarray as xr
import numpy as np


COORD_RENAME_INVERSE_MAP = {
    "x": {"grid_xt", "grid_xt_coarse"},
    "y": {"grid_yt", "grid_yt_coarse"},
    "tile": {"rank"},
}


def _adjust_tile_range(ds: xr.Dataset) -> xr.Dataset:

    if "tile" in ds:
        tiles = ds.tile

        if tiles.isel(tile=-1) == 6:
            ds = ds.assign_coords({"tile": tiles - 1})

    return ds


def _rename_coords(ds: xr.Dataset) -> xr.Dataset:

    varname_target_registry = {}
    for target_name, source_names in COORD_RENAME_INVERSE_MAP.items():
        varname_target_registry.update({name: target_name for name in source_names})

    vars_to_rename = {
        var: varname_target_registry[var] 
        for var in ds.coords if var in varname_target_registry
    }
    ds = ds.rename(vars_to_rename)
    return ds


def _round_microseconds(dt):
    inc = datetime.timedelta(seconds=round(dt.microsecond * 1e-6))
    dt = dt.replace(microsecond=0)
    dt += inc
    return dt


def _round_time_coord(ds, time_coord="time"):
    
    new_times = np.vectorize(_round_microseconds)(ds.time)
    ds = ds.assign_coords({time_coord: new_times})
    return ds


def _set_missing_attrs(ds):

    for var in ds:
        da = ds[var]

        # True for some prognostic zarrs
        if "description" in da.attrs and "long_name" not in da.attrs:
            da.attrs["long_name"] = da.attrs["description"]
        
        if "long_name" not in da.attrs:
            da.attrs["long_name"] = var
        
        if "units" not in da.attrs:
            da.attrs["units"] = "unspecified"
    return ds


def _open_tiles(path):
    return xr.open_mfdataset(path + ".tile?.nc", concat_dim="tile", combine="nested")


def load_verification():

    atmos = catalog["40day_c384_atmos_8xdaily"].to_dask() #3H
    sfc_diags = catalog["40day_c384_diags_time_avg"].to_dask() #15min


