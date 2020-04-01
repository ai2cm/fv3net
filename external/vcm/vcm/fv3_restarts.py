import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Generator, Tuple

import xarray as xr
from dask.delayed import delayed

from vcm.cloud.fsspec import get_fs
from vcm.combining import combine_array_sequence
from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_X_OUTER,
    COORD_Y_CENTER,
    COORD_Y_OUTER,
    FV_CORE_X_CENTER,
    FV_CORE_X_OUTER,
    FV_CORE_Y_CENTER,
    FV_CORE_Y_OUTER,
    FV_SRF_WND_X_CENTER,
    FV_SRF_WND_Y_CENTER,
    FV_TRACER_X_CENTER,
    FV_TRACER_Y_CENTER,
    RESTART_Z_CENTER,
    SFC_DATA_X_CENTER,
    SFC_DATA_Y_CENTER,
    TILE_COORDS_FILENAMES,
    COORD_Z_CENTER,
    COORD_Z_SOIL,
    COORD_Z_OUTER,
    RESTART_Z_CENTER
)
from vcm.schema_registry import impose_dataset_to_schema
from vcm.xarray_loaders import open_delayed

from . import _rundir
from .cubedsphere import constants

SCHEMA_CACHE = {}
FILE_PREFIX_DIM = "file_prefix"

RESTART_CATEGORIES = {
    "core": ["u", "v", "W", "DZ", "T", "delp", "phis"],
    "srf_wnd": ["u_srf", "v_srf"],
    "tracer": [
        "sphum",
        "liq_wat",
        "rainwat",
        "ice_wat",
        "snowwat",
        "graupel",
        "o3mr",
        "cld_amt",
    ],
    "sfc": [
        "slmsk",
        "tsea",
        "sheleg",
        "tg3",
        "zorl",
        "alvsf",
        "alvwf",
        "alnsf",
        "alnwf",
        "facsf",
        "facwf",
        "vfrac",
        "canopy",
        "f10m",
        "t2m",
        "q2m",
        "vtype",
        "stype",
        "uustar",
        "ffmm",
        "ffhh",
        "hice",
        "fice",
        "tisfc",
        "tprcp",
        "srflag",
        "snwdph",
        "shdmin",
        "shdmax",
        "slope",
        "snoalb",
        "sncovr",
        "stc",
        "smc",
        "slc",
    ],
}


CATEGORY_OF_VARIABLE = {
    name: category
    for category in RESTART_CATEGORIES
    for name in RESTART_CATEGORIES[category]
}

FV_CORE_DIMS = {
    COORD_X_CENTER: FV_CORE_X_CENTER,
    COORD_Y_CENTER: FV_CORE_Y_CENTER,
    COORD_X_OUTER: FV_CORE_X_OUTER,
    COORD_Y_OUTER: FV_CORE_Y_OUTER,
    COORD_Z_CENTER: "zaxis_1",
}

FV_TRACER_DIMS = {
    COORD_X_CENTER: FV_TRACER_X_CENTER,
    COORD_Y_CENTER: FV_TRACER_Y_CENTER,
    COORD_Z_CENTER: "zaxis_1",
}

SFC_DATA_DIMS = {
    COORD_X_CENTER: SFC_DATA_X_CENTER,
    COORD_Y_CENTER: SFC_DATA_Y_CENTER,
    COORD_Z_SOIL: "zaxis_1",
}

FV_SRF_WND_DIMS = {
    COORD_X_CENTER: FV_SRF_WND_X_CENTER,
    COORD_Y_CENTER: FV_SRF_WND_Y_CENTER,
}


def open_diagnostic(url, category):
    fs = get_fs(url)
    diag_tiles = []
    for tile in TILE_COORDS_FILENAMES:
        tile_file = f"{category}.tile{tile}.nc"
        with fs.open(os.path.join(url, tile_file), "rb") as f:
            diag_tiles.append(xr.open_dataset(f))
    return xr.concat(diag_tiles, "tile")


def open_restarts(url: str) -> xr.Dataset:
    """Opens all the restart file within a certain path

    The dimension names are the same as the diagnostic output

    Args:
        url (str): a URL to the root directory of a run directory.
            Can be any type of protocol used by fsspec, such as google cloud storage
            'gs://path-to-rundir'. If no protocol prefix is used, then it will be
            assumed to be a path to a local file.

    Returns:
        ds (xr.Dataset): a combined dataset of all the restart files. All except
            the first file of each restart-file type (e.g. fv_core.res) will only
            be lazily loaded. This allows opening large datasets out-of-core.

    """
    fs = get_fs(url)
    walker = fs.walk(url)
    restart_files = _rundir.yield_restart_files(walker)
    arrays = _load_arrays(fs, restart_files)
    return xr.Dataset(combine_array_sequence(arrays, labels=[FILE_PREFIX_DIM, "tile"]))


def open_restarts_with_time_coordinates(url: str) -> xr.Dataset:
    """Opens all the restart file within a certain path, with time coordinates

    The dimension names are the same as the diagnostic output

    Args:
        url (str): a URL to the root directory of a run directory.
            Can be any type of protocol used by fsspec, such as google cloud storage
            'gs://path-to-rundir'. If no protocol prefix is used, then it will be
            assumed to be a path to a local file.

    Returns:
        ds (xr.Dataset): a combined dataset of all the restart files. All except
            the first file of each restart-file type (e.g. fv_core.res) will only
            be lazily loaded. This allows opening large datasets out-of-core.
            Time coordinates are inferred from the run directory's namelist and
            other files.

    """
    time_dim = "time"

    fs = get_fs(url)
    ds = open_restarts(url)
    mapping = _rundir.get_prefix_time_mapping(fs, url)
    ds_with_times = _replace_1d_coord_by_mapping(ds, mapping, FILE_PREFIX_DIM, time_dim)
    return ds_with_times.sortby(time_dim)


def standardize_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Update the meta-data of an individual restart file

    This drops the singleton time dimension and applies the known dimensions
    listed in `vcm.schema` and `vcm._schema_registry`.
    """
    try:
        ds_no_time = ds.isel(Time=0).drop("Time")
    except ValueError:
        ds_no_time = ds
    return impose_dataset_to_schema(ds_no_time)


RestartCategories = namedtuple(
    "RestartCategories", ("core", "tracer", "srf_wnd", "sfc")
)


def split_into_restart_categories(restart: xr.Dataset) -> RestartCategories:
    categories = {}

    for category, variable_list in RESTART_CATEGORIES.items():
        categories[category] = restart[
            [variable for variable in variable_list if variable in restart]
        ]

    return RestartCategories(
        categories["core"],
        categories["tracer"],
        categories["srf_wnd"],
        categories["sfc"],
    )


def _rename_restart_dims(data: RestartCategories) -> RestartCategories:
    return RestartCategories(
        data.core.rename(FV_CORE_DIMS),
        data.tracer.rename(FV_TRACER_DIMS),
        data.srf_wnd.rename(FV_SRF_WND_DIMS),
        data.sfc.rename(SFC_DATA_DIMS),
    )


def _save_category(ds: xr.Dataset, name, output_dir):
    for tile in range(6):
        tile_data = ds.isel(tile=tile)
        path = os.path.join(output_dir, name + ".tile{tile+1}.nc")
        tile_data.to_netcdf(path)


def to_restart_netcdfs(ds: xr.Dataset, output_dir: str):
    """Save a single set of combined restart data as FV3 restart files

    Args
        ds: single set of restart data with standardized names
        output_dir: a local output directory.

    Notes:
        remote output is not needed because this function should 
        only be used when running fv3 locally.
    """
    restart = split_into_restart_categories(ds)
    renamed = _rename_restart_dims(restart)
    for name, data in [
        ("fv_core.res", renamed.core),
        ("fv_tracer.res", renamed.tracer),
        ("sfc_data", renamed.sfc),
        ("fv_srf_wnd", renamed.srf_wnd),
    ]:
        for tile in range(6):
            tile_data = data.isel(tile=tile)
            path = os.path.join(output_dir, name + f".tile{tile+1}.nc")
            tile_data.to_netcdf(path)


def _replace_1d_coord_by_mapping(ds, mapping, old_dim, new_dim="time"):
    coord = ds[old_dim]
    times = xr.DataArray([mapping[prefix.item()] for prefix in coord], dims=[old_dim])
    return ds.assign_coords({new_dim: times}).swap_dims({old_dim: new_dim})


def _load_restart(fs, path):
    with fs.open(path) as f:
        return xr.open_dataset(f).compute()


def _load_restart_with_schema(fs, path, schema):
    promise = delayed(_load_restart)(fs, path)
    return open_delayed(promise, schema)


def _load_restart_lazily(fs, path, restart_category):
    # only actively load the initial data
    if restart_category in SCHEMA_CACHE:
        schema = SCHEMA_CACHE[restart_category]
    else:
        schema = _load_restart(fs, path)
        SCHEMA_CACHE[restart_category] = schema

    return _load_restart_with_schema(fs, path, schema)


def _load_arrays(
    fs, restart_files
) -> Generator[Tuple[Any, Tuple, xr.DataArray], None, None]:
    # use the same schema for all coupler_res
    for (file_prefix, restart_category, tile, path) in restart_files:
        ds = _load_restart_lazily(fs, path, restart_category)
        ds_standard_metadata = standardize_metadata(ds)
        for var in ds_standard_metadata:
            yield var, (file_prefix, tile), ds_standard_metadata[var]
