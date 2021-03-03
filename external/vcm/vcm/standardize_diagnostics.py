import xarray as xr
from typing import Iterable, Mapping, Set
import warnings
import logging

logger = logging.getLogger(__name__)

# desired name as keys with set containing sources to rename
# TODO: could this be tied to the registry?
DIM_RENAME_INVERSE_MAP = {
    "x": {"grid_xt", "grid_xt_coarse"},
    "y": {"grid_yt", "grid_yt_coarse"},
    "tile": {"rank"},
    "xb": {"grid_x", "grid_x_coarse", "x_interface"},
    "yb": {"grid_y", "grid_y_coarse", "y_interface"},
}
VARNAME_SUFFIX_TO_REMOVE = ["_coarse"]


def standardize_gfsphysics_diagnostics(ds):

    for func in [
        _set_calendar_to_julian,
        xr.decode_cf,
        _adjust_tile_range,
        _rename_dims,
        _round_time_coord,
        _remove_name_suffix,
        _set_missing_attrs,
    ]:
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
        var: varname_target_registry[var]
        for var in ds.dims
        if var in varname_target_registry
    }
    ds = ds.rename(vars_to_rename)
    return ds


def _set_calendar_to_julian(ds, time_coord="time"):
    if time_coord in ds.coords:
        ds[time_coord].attrs["calendar"] = "julian"
    return ds


def _round_to_nearest_second(time: xr.DataArray) -> xr.DataArray:
    return time.dt.round("1S")


def _round_time_coord(ds, time_coord="time"):

    if time_coord in ds.coords:
        new_times = _round_to_nearest_second(ds[time_coord])
        ds = ds.assign_coords({time_coord: new_times})
    else:
        logger.debug(
            "Round time operation called on dataset missing a time coordinate."
        )

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


def _remove_name_suffix(ds):
    for target in VARNAME_SUFFIX_TO_REMOVE:
        replace_names = {
            vname: vname.replace(target, "")
            for vname in ds.data_vars
            if target in vname
        }

        _warn_on_overwrite(ds.data_vars.keys(), replace_names.values())
        ds = ds.rename(replace_names)
    return ds


def _warn_on_overwrite(old: Iterable, new: Iterable):
    """
    Warn if new data keys will overwrite names (e.g., in a xr.Dataset)
    via an overlap with old keys or from duplication in new keys.

    Args:
        old: Original keys to check against
        new: Incoming keys to check for duplicates or existence in old
    """
    duplicates = {item for item in new if list(new).count(item) > 1}
    overlap = set(old) & set(new)
    overwrites = duplicates | overlap
    if len(overwrites) > 0:
        warnings.warn(
            UserWarning(
                f"Overwriting keys detected. Overlap: {overlap}"
                f"  Duplicates: {duplicates}"
            )
        )
