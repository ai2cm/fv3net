import intake
import logging
import warnings
import os
import xarray as xr
import numpy as np
from typing import List, Iterable, Tuple, Mapping, Set
from datetime import timedelta

import fsspec
import vcm

logger = logging.getLogger(__name__)

DiagArg = Tuple[xr.Dataset, xr.Dataset, xr.Dataset]

# desired name as keys with set containing sources to rename
# TODO: could this be tied to the registry?
DIM_RENAME_INVERSE_MAP = {
    "x": {"grid_xt", "grid_xt_coarse"},
    "y": {"grid_yt", "grid_yt_coarse"},
    "tile": {"rank"},
    "xb": {"grid_x", "grid_x_coarse"},
    "yb": {"grid_y", "grid_y_coarse"},
}
VARNAME_SUFFIX_TO_REMOVE = ["_coarse"]
_DIAG_OUTPUT_LOADERS = []


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


def _round_to_nearest_second(dt):
    return vcm.convenience.round_time(dt, timedelta(seconds=1))


def _round_time_coord(ds, time_coord="time"):

    if time_coord in ds.coords:
        new_times = np.vectorize(_round_to_nearest_second)(ds.time)
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
        _adjust_tile_range,
        _rename_dims,
        _round_time_coord,
        _remove_name_suffix,
        _set_missing_attrs,
    ]:
        ds = func(ds)

    return ds


def _open_tiles(path):
    return xr.open_mfdataset(path + ".tile?.nc", concat_dim="tile", combine="nested")


def load_verification(
    catalog_keys: List[str],
    catalog: intake.Catalog,
    coarsening_factor: int = None,
    area: xr.DataArray = None,
) -> xr.Dataset:

    """
    Load verification data sources from a catalog and combine for reporting.

    Args:
        catalog_keys: catalog sources to load as verification data
        catalog: Intake catalog of available data sources.
        coarsening_factor (optional): Factor to coarsen the loaded verification data
        area (optional): Grid cell area data for weighting. Required when
            coarsening_factor is set.

    Returns:
        All specified verification datasources standardized and merged

    """

    area = _rename_dims(area)

    verif_data = []
    for dataset_key in catalog_keys:
        ds = catalog[dataset_key].to_dask()
        ds = standardize_gfsphysics_diagnostics(ds)

        if coarsening_factor is not None:
            if area is None:
                raise ValueError(
                    "Grid area keyword argument must be provided when"
                    " coarsening is requested."
                )

            ds = vcm.cubedsphere.weighted_block_average(
                ds, area, coarsening_factor, x_dim="x", y_dim="y"
            )

        verif_data.append(ds)

    return xr.merge(verif_data, join="outer")


def _load_standardized(path):
    logger.info(f"Loading and standardizing {path}")
    m = fsspec.get_mapper(path)
    ds = xr.open_zarr(m, consolidated=True).load()
    return standardize_gfsphysics_diagnostics(ds)


def _load_prognostic_run_physics_output(url):
    """Load, standardize and merge prognostic run physics outputs"""
    # values euqal to zero in diags.zarr get interpreted as nans by xarray, so fix here
    diagnostic_data = [
        _load_standardized(os.path.join(url, "diags.zarr")).fillna(0.0),
        _load_standardized(os.path.join(url, "sfc_dt_atmos.zarr")),
    ]

    # TODO: diags.zarr currently doesn't contain any coordinates and should perhaps be
    #       remedied. Need to handle crashed run extra timestep in here for now.
    cutoff_time_index = min([len(ds["time"]) for ds in diagnostic_data])
    diagnostic_data = [
        ds.isel(time=slice(0, cutoff_time_index)) for ds in diagnostic_data
    ]

    return xr.merge(diagnostic_data, join="inner")


def load_dycore(url: str, grid_spec: str, catalog: intake.Catalog) -> DiagArg:
    """Open data required for dycore plots.

    Args:
        url: path to prognostic run directory
        grid_spec: path to C384 grid spec (everything up to .tile?.nc)
        catalog: Intake catalog of available data sources

    Returns:
        tuple of prognostic run data, verification data and grid variables all at
        coarsened resolution. Prognostic and verification data contain variables output
        by the dynamical core.
    """
    logger.info(f"Processing dycore data from run directory at {url}")

    # open grid
    logger.info("Opening Grid Spec")
    grid_c384 = standardize_gfsphysics_diagnostics(vcm.open_tiles(grid_spec))
    grid_c48 = vcm.cubedsphere.weighted_block_average(
        grid_c384, grid_c384.area, 8, x_dim="x", y_dim="y"
    )

    # open verification
    logger.info("Opening verification data")
    verification_c48 = load_verification(
        ["40day_c384_atmos_8xdaily"], catalog, coarsening_factor=8, area=grid_c384.area
    )

    # open prognostic run data
    path = os.path.join(url, "atmos_dt_atmos.zarr")
    logger.info(f"Opening prognostic run data at {path}")
    ds = _load_standardized(path)
    resampled = ds.resample(time="3H", label="right").nearest()

    # don't use last time point. there is some trouble
    resampled = resampled.isel(time=slice(None, -1))
    verification_c48 = verification_c48.sel(time=resampled.time)

    return resampled, verification_c48, grid_c48[["area"]]


def load_physics(url: str, grid_spec: str, catalog: intake.Catalog) -> DiagArg:
    """Open data required for physics plots.

        Args:
            url: path to prognostic run directory
            grid_spec: path to C384 grid spec (everything up to .tile?.nc)
            catalog: Intake catalog of available data sources

        Returns:
            tuple of prognostic run data, verification data and grid variables all at
            coarsened resolution. Prognostic and verification data contain variables
            output by the physics routines.
        """
    logger.info(f"Processing physics data from run directory at {url}")

    # open grid
    logger.info("Opening Grid Spec")
    grid_c384 = standardize_gfsphysics_diagnostics(vcm.open_tiles(grid_spec))
    grid_c48 = vcm.cubedsphere.weighted_block_average(
        grid_c384, grid_c384.area, 8, x_dim="x", y_dim="y"
    )

    # open verification
    # TODO: load physics diagnostics for SHiELD data. Currently slow due to chunking.
    verification_c48 = xr.Dataset()

    # open prognostic run data
    logger.info(f"Opening prognostic run data at {url}")
    prognostic_output = _load_prognostic_run_physics_output(url)
    # resample since 15-minute frequency timeseries makes report bloated
    resampled = prognostic_output.resample(time="3H", label="right").nearest()

    # avoid last time point since it often distorts plot limits
    resampled = resampled.isel(time=slice(None, -1))

    return resampled, verification_c48, grid_c48[["area"]]
