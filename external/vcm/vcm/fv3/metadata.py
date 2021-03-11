import xarray as xr
from typing import Mapping, Set, Callable, Sequence
import logging
from ..convenience import round_time
from ..safe import warn_if_intersecting

logger = logging.getLogger(__name__)


DIM_RENAME_INVERSE_MAP = {
    "x": {"grid_xt", "grid_xt_coarse"},
    "y": {"grid_yt", "grid_yt_coarse"},
    "tile": {"rank"},
    "x_interface": {"grid_x", "grid_x_coarse"},
    "y_interface": {"grid_y", "grid_y_coarse"},
}
VARNAME_SUFFIX_TO_REMOVE = ["_coarse"]
TIME_DIM_NAME = "time"
STANDARD_TO_GFDL_DIM_MAP = {
    "x_interface": "grid_x",
    "y_interface": "grid_y",
    "x": "grid_xt",
    "y": "grid_yt",
    "z": "pfull",
}


def standardize_fv3_diagnostics(
    ds: xr.Dataset, time: str = TIME_DIM_NAME
) -> xr.Dataset:
    """Standardize dimensions, coordinates, and attributes of FV3 diagnostic output
    
    Args:
        ds (xr.Dataset):
            Dataset of FV3GFS or SHiELD diagnostics outputs (either Fortran or
            python-derived), presumably opened from disk via zarr.
        time (str, optional):
            Name of the time coordinate dimension in the dataset
        
    Returns:
        The dataset with coordinate names and tile ranges set to CF "standard" values,
        variable name suffixes (e.g., "_coarse") removed, and data variable attributes
        ("long_name" and "units") set. The time coordinate is rounded to the nearest
        second.
    
    """

    if time in ds.coords:
        ds[time].attrs["calendar"] = "julian"

    funcs: Sequence[Callable[[xr.Dataset], xr.Dataset]] = [
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


def _round_time_coord(ds: xr.Dataset, time_coord: str = TIME_DIM_NAME) -> xr.Dataset:

    if time_coord in ds.coords:
        new_times = round_time(ds[time_coord])
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

        warn_if_intersecting(ds.data_vars.keys(), replace_names.values())
        ds = ds.rename(replace_names)
    return ds


def gfdl_to_standard(ds: xr.Dataset):
    """Convert from GFDL dimension names (grid_xt, etc) to "standard"
    names (x, y, z)
    """

    key, val = STANDARD_TO_GFDL_DIM_MAP.keys(), STANDARD_TO_GFDL_DIM_MAP.values()
    inverse = dict(zip(val, key))

    return ds.rename({key: val for key, val in inverse.items() if key in ds.dims})


def standard_to_gfdl(ds: xr.Dataset):
    """Convert from "standard" names to GFDL names"""
    return ds.rename(
        {key: val for key, val in STANDARD_TO_GFDL_DIM_MAP.items() if key in ds.dims}
    )
