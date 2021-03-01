import logging
import warnings
import xarray as xr
import intake
from vcm.catalog import catalog as CATALOG
from typing import (
    Hashable,
    Mapping,
    Any,
    MutableMapping,
    Set,
    Iterable,
)

from ._nudged._legacy import MergeNudged
from ._base import LongRunMapper
from ._nudged import open_nudge_to_fine

logger = logging.getLogger(__name__)

Z_DIM_NAME = "z"

Time = str
Dataset = MutableMapping[Hashable, Any]


def open_n2f_radiative_flux_biases(
    url: str,
    verification: str = "40day_c48_gfsphysics_15min_may2020",
    consolidated: bool = True,
    **open_nudge_to_fine_kwargs: Mapping[str, Any],
):
    """
    Include surface radiative flux biases.
    Merges variables saved in the model state and fine- and coarse-resolution
    sfc_dt_atmos zarrs.
    
    Args:
        url (str):  path to nudge-to-fine output directory, remote or local
        verification (str): Name of entry in catalog to use as verification dataset
        consolidated (bool): whether zarrs to open have consolidated metadata
        **open_nudge_to_fine_kwargs: kwargs to be passed to `open_nudge_to_fine`
        
    Returns:
        mapper to LW and SW surface radiative flux biases and model state data
    """
    verif_dswrf_sfc, verif_swnetrf_sfc, verif_dlwrf_sfc = _get_verification_fluxes(
        CATALOG, verification
    )
    verif_dswrf_sfc_mean = verif_dswrf_sfc.mean(dim="time")
    surface_albedo = (
        verif_dswrf_sfc_mean - verif_swnetrf_sfc.mean(dim="time")
    ) / verif_dswrf_sfc_mean
    mean_albedo = surface_albedo.mean().values
    surface_albedo = (
        surface_albedo.fillna(mean_albedo)
        .broadcast_like(verif_dswrf_sfc)
        .assign_attrs({"long_name": "surface albedo (coarse)", "units": "-"})
    )
    sfc_biases_ds = xr.Dataset(
        {
            "DSWRFsfc_verif": verif_dswrf_sfc,
            "NSWRFsfc_verif": verif_swnetrf_sfc,
            "DLWRFsfc_verif": verif_dlwrf_sfc,
            "surface_albedo": surface_albedo,
        }
    )
    sfc_biases_mapper = LongRunMapper(sfc_biases_ds)
    nudge_to_fine_mapper = open_nudge_to_fine(
        url, **open_nudge_to_fine_kwargs, consolidated=consolidated
    )
    return MergeNudged(nudge_to_fine_mapper, sfc_biases_mapper)


def _get_verification_fluxes(catalog: intake.catalog, verification: str) -> xr.Dataset:
    ds = catalog[verification].to_dask()
    ds = standardize_gfsphysics_diagnostics(ds)
    swnetrf_sfc = (ds["DSWRFsfc"] - ds["USWRFsfc"]).assign_attrs(
        {"long_name": "net shortwave surface flux (down minus up)", "units": "W/m^2"}
    )
    return ds["DSWRFsfc"], swnetrf_sfc, ds["DLWRFsfc"]


# note that the below is cut/pasted from prognostic_run_diags workflow
# TODO: refactor to a common location if we want to keep using this


DIM_RENAME_INVERSE_MAP = {
    "x": {"grid_xt", "grid_xt_coarse"},
    "y": {"grid_yt", "grid_yt_coarse"},
    "tile": {"rank"},
    "xb": {"grid_x", "grid_x_coarse", "x_interface"},
    "yb": {"grid_y", "grid_y_coarse", "y_interface"},
}
VARNAME_SUFFIX_TO_REMOVE = ["_coarse"]


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

        warn_on_overwrite(ds.data_vars.keys(), replace_names.values())
        ds = ds.rename(replace_names)
    return ds


def warn_on_overwrite(old: Iterable, new: Iterable):
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
