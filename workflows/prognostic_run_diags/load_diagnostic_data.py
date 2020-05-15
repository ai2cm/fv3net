import intake
import xarray as xr


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


def load_verification():

    atmos = catalog["40day_c384_atmos_8xdaily"].todask() #3H
    sfc_diags = catalog["40day_c384_diags_time_avg"].todask() #15min
