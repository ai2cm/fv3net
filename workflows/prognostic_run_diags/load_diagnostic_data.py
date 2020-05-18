import intake
import datetime
import logging
import warnings
import os
import xarray as xr
import numpy as np
from typing import List, Iterable
from pathlib import Path

import vcm

logger = logging.getLogger(__name__)

# desired name as keys with set containing sources to rename
# TODO: could this be tied to the registry?
COORD_RENAME_INVERSE_MAP = {
    "x": {"grid_xt", "grid_xt_coarse"},
    "y": {"grid_yt", "grid_yt_coarse"},
    "tile": {"rank"},
}
VARNAME_SUFFIX_TO_REMOVE = ["_coarse"]

_DS_TRANSFORMS = []
_DIAG_OUTPUT_LOADERS = []


def add_to_transforms(func):
    """
    Add to xr.Dataset transform function to the group of
    transforms to be performed on a loaded dataset.
    
    Args:
        func: A functions which adjusts an xr.Datset.
            It needs to have the following signature::

                func(ds: xr.Dataset)

            and should return an xarray Dataset.
    """

    _DS_TRANSFORMS.append(func)
    return func


@add_to_transforms
def _adjust_tile_range(ds: xr.Dataset) -> xr.Dataset:

    if "tile" in ds:
        tiles = ds.tile

        if tiles.isel(tile=-1) == 6:
            ds = ds.assign_coords({"tile": tiles - 1})

    return ds


@add_to_transforms
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


@add_to_transforms
def _round_time_coord(ds, time_coord="time"):
    
    new_times = np.vectorize(_round_microseconds)(ds.time)
    ds = ds.assign_coords({time_coord: new_times})
    return ds


@add_to_transforms
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


@add_to_transforms
def _remove_name_suffix(ds):
    for target in VARNAME_SUFFIX_TO_REMOVE:
        replace_names = {vname: vname.replace(target, "")
                         for vname in ds.data_vars if target in vname}

        warn_on_overwrite(replace_names.data_vars.keys(), replace_names.values())
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


def standardize_dataset(ds):

    for func in _DS_TRANSFORMS:
        ds = func(ds)

    return ds


def _open_tiles(path):
    return xr.open_mfdataset(path + ".tile?.nc", concat_dim="tile", combine="nested")


def _catalog():
    TOP_LEVEL_DIR = Path(os.path.abspath(__file__)).parent.parent.parent
    path = str(TOP_LEVEL_DIR / "catalog.yml")
    return intake.open_catalog(path)


def load_verification(
    catalog_keys: List[str],
    catalog: intake.Catalog = None,
    coarsen_factor: int = None,
    area: xr.DataArray = None
) -> xr.Dataset:

    """
    Load verification data sources from a catalog and combine for reporting.

    Args:
        catalog_keys: catalog sources to load as verification data
        catalog (optional): Intake catalog of available data sources.  Defaults
            to fv3net top-level "catalog.yml" catalog.
        coarsen_factor (optional): Factor to coarsen the loaded verification data
        area (optional): Grid cell area data for weighting. Required when 
            coarsen_factor is set.

    Returns:
        All specified verification datasources standardized and merged

    """

    if catalog is None:
        catalog = _catalog()

    verif_data = []
    for dataset_key in catalog_keys:
        ds = catalog[dataset_key].to_dask()
        ds = standardize_dataset(ds)
        
        if coarsen_factor is not None:
            if area is None:
                raise ValueError("Grid area keyword argument must be provided when"
                                 " coarsening is requested.")

            ds = vcm.cubedsphere.weighted_block_average(
                ds, area, coarsen_factor, x_dim="x", y_dim="y"
            )

    return xr.merge(verif_data, join="outer")


def add_to_diag_loaders(func):
    """
    Add diagnostic output file loader to the group of diagnostic
    datasets to combine from an FV3GFS run path.
    
    Args:
        func: A functions which loads an xr.Datset.
            It needs to have the following signature::

                func(path: str)

            and should return an xarray Dataset.
    """

    _DIAG_OUTPUT_LOADERS.append(func)
    return func


@add_to_diag_loaders
def _load_diags_zarr(url):

    pass


def load_diagnostics(url):
    pass
