import xarray as xr
from typing import Mapping, Set, Callable, Sequence
import logging
from .convenience import warn_on_overwrite

logger = logging.getLogger(__name__)


DIM_RENAME_INVERSE_MAP = {
    "x": {"grid_xt", "grid_xt_coarse"},
    "y": {"grid_yt", "grid_yt_coarse"},
    "tile": {"rank"},
    "xb": {"grid_x", "grid_x_coarse", "x_interface"},
    "yb": {"grid_y", "grid_y_coarse", "y_interface"},
}
VARNAME_SUFFIX_TO_REMOVE = ["_coarse"]


def standardize_gfsphysics_diagnostics(ds: xr.Dataset) -> xr.Dataset:

    funcs: Sequence[Callable[[xr.Dataset], xr.Dataset]] = [
        _set_calendar_to_julian,
        xr.decode_cf,
        _adjust_tile_range,
        _rename_dims,
        _round_time_coord,
        _remove_name_suffix,
        _set_missing_attrs,
    ]

    for func in funcs:
        ds = func(ds)

    return ds


def _adjust_tile_range(ds: xr.Dataset) -> xr.Dataset:

    if "tile" in ds:
        tiles = ds.tile

        if tiles.isel(tile=-1) == 6:
            ds = ds.assign_coords({"tile": tiles - 1})

    return ds


def _rename_dims(
    ds: xr.Dataset, rename_inverse: Mapping[str, Set[str]] = DIM_RENAME_INVERSE_MAP
) -> xr.Dataset:

    varname_target_registry = {}
    for target_name, source_names in rename_inverse.items():
        varname_target_registry.update({name: target_name for name in source_names})

    vars_to_rename = {
        var: varname_target_registry[str(var)]
        for var in ds.dims
        if var in varname_target_registry
    }
    ds = ds.rename(vars_to_rename)
    return ds


def _set_calendar_to_julian(ds: xr.Dataset, time_coord: str = "time") -> xr.Dataset:
    if time_coord in ds.coords:
        ds[time_coord].attrs["calendar"] = "julian"
    return ds


def _round_to_nearest_second(time: xr.DataArray) -> xr.DataArray:
    return time.dt.round("1S")


def _round_time_coord(ds: xr.Dataset, time_coord: str = "time") -> xr.Dataset:

    if time_coord in ds.coords:
        new_times = _round_to_nearest_second(ds[time_coord])
        ds = ds.assign_coords({time_coord: new_times})
    else:
        logger.debug(
            "Round time operation called on dataset missing a time coordinate."
        )

    return ds


def _set_missing_attrs(ds: xr.Dataset) -> xr.Dataset:

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


def _remove_name_suffix(
    ds: xr.Dataset, suffixes: Sequence[str] = VARNAME_SUFFIX_TO_REMOVE
) -> xr.Dataset:
    for target in suffixes:
        replace_names = {
            vname: str(vname).replace(target, "")
            for vname in ds.data_vars
            if target in str(vname)
        }

        warn_on_overwrite(ds.data_vars.keys(), replace_names.values())
        ds = ds.rename(replace_names)
    return ds
