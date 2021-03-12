import intake
import logging
import numpy as np
import os
import xarray as xr
from typing import List, Mapping, Sequence

import fsspec
import vcm
from vcm.cloud import get_fs
from vcm.fv3 import standardize_fv3_diagnostics
from fv3net.diagnostics.prognostic_run import add_derived
from fv3net.diagnostics.prognostic_run.constants import DiagArg

logger = logging.getLogger(__name__)


GRID_ENTRIES = {48: "grid/c48", 96: "grid/c96", 384: "grid/c384"}


def load_verification(catalog_keys: List[str], catalog: intake.Catalog,) -> xr.Dataset:

    """
    Load verification data sources from a catalog and combine for reporting.

    Args:
        catalog_keys: catalog sources to load as verification data
        catalog: Intake catalog of available data sources.

    Returns:
        All specified verification datasources standardized and merged

    """
    verif_data = []
    for dataset_key in catalog_keys:
        ds = catalog[dataset_key].to_dask()
        ds = standardize_fv3_diagnostics(ds)
        verif_data.append(ds)
    return xr.merge(verif_data, join="outer")


def _load_standardized(path):
    logger.info(f"Loading and standardizing {path}")
    m = fsspec.get_mapper(path)
    ds = xr.open_zarr(m, consolidated=True, decode_times=False)
    return standardize_fv3_diagnostics(ds)


def _load_prognostic_run_physics_output(url):
    """Load, standardize and merge prognostic run physics outputs"""
    diags_url = os.path.join(url, "diags.zarr")
    sfc_dt_atmos_url = os.path.join(url, "sfc_dt_atmos.zarr")
    diagnostic_data = [_load_standardized(sfc_dt_atmos_url)]
    try:
        diags_ds = _load_standardized(diags_url)
    except (FileNotFoundError, KeyError):
        # don't fail if diags.zarr doesn't exist (fsspec raises KeyError)
        pass
    else:
        # values equal to zero in diags.zarr may get interpreted as nans by xarray
        diagnostic_data.append(diags_ds.fillna(0.0))
    return xr.merge(diagnostic_data, join="inner")


def _coarsen(ds: xr.Dataset, area: xr.DataArray, coarsening_factor: int) -> xr.Dataset:
    return vcm.cubedsphere.weighted_block_average(
        ds, area, coarsening_factor, x_dim="x", y_dim="y"
    )


def _get_coarsening_args(
    ds: xr.Dataset, target_res: int, grid_entries: Mapping[int, str] = GRID_ENTRIES
) -> (str, int):
    """Given input dataset and target resolution, return catalog entry for input grid
    and coarsening factor"""
    input_res = ds.sizes["x"]
    if input_res % target_res != 0:
        raise ValueError("Target resolution must evenly divide input resolution")
    coarsening_factor = int(input_res / target_res)
    if input_res not in grid_entries:
        raise KeyError(f"No grid defined in catalog for c{input_res} resolution")
    return grid_entries[input_res], coarsening_factor


def _load_prognostic_run_3d_output(url: str):
    fs = get_fs(url)
    prognostic_3d_output = [item for item in fs.ls(url) if item.endswith("xdaily.zarr")]
    if len(prognostic_3d_output) > 0:
        zarr_name = os.path.basename(prognostic_3d_output[0])
        path = os.path.join(url, zarr_name)
        return _load_standardized(path)
    else:
        return None


def load_3d(
    url: str, verification_entries: Sequence[str], catalog: intake.Catalog
) -> DiagArg:
    logger.info(f"Processing 3d data from run directory at {url}")

    # open prognostic run data. If 3d data not saved, return empty datasets.
    ds = _load_prognostic_run_3d_output(url)
    if ds is None:
        return xr.Dataset(), xr.Dataset(), xr.Dataset()

    else:
        input_grid, coarsening_factor = _get_coarsening_args(ds, 48)
        area = catalog[input_grid].to_dask()["area"]
        ds = _coarsen(ds, area, coarsening_factor)

        # open grid
        logger.info("Opening Grid Spec")
        grid_c48 = standardize_fv3_diagnostics(catalog["grid/c48"].to_dask())

        # rename to common variable names
        renamed = {
            "temp": "air_temperature",
            "w": "vertical_wind",
            "sphum": "specific_humidity",
            "ucomp": "eastward_wind",
            "vcomp": "northward_wind",
        }
        ds = ds.rename(renamed)

        # interpolate 3d prognostic fields to pressure levels
        ds_interp = xr.Dataset()
        for var in renamed.values():
            ds_interp[var] = vcm.interpolate_to_pressure_levels(
                field=ds[var], delp=ds["delp"]
            )

        # open verification
        logger.info("Opening verification data")
        verification_c48 = load_verification(verification_entries, catalog)

        # Not all verification datasets have 3D variables saved,
        # if not available fill with NaNs
        if len(verification_c48.data_vars) == 0:
            for var in ds_interp:
                verification_c48[var] = xr.full_like(ds_interp[var], np.nan)
                verification_c48[var].attrs = ds_interp[var].attrs
        return ds_interp, verification_c48, grid_c48


def load_dycore(
    url: str, verification_entries: Sequence[str], catalog: intake.Catalog
) -> DiagArg:
    """Open data required for dycore plots.

    Args:
        url: path to prognostic run directory
        verification_entries: catalog entries for verification dycore data
        catalog: Intake catalog of available data sources

    Returns:
        tuple of prognostic run data, verification data and grid variables all at
        coarsened resolution. Prognostic and verification data contain variables output
        by the dynamical core.
    """
    logger.info(f"Processing dycore data from run directory at {url}")

    # open grid
    logger.info("Opening Grid Spec")
    grid_c48 = standardize_fv3_diagnostics(catalog["grid/c48"].to_dask())
    ls_mask = standardize_fv3_diagnostics(catalog["landseamask/c48"].to_dask())
    grid_c48 = xr.merge([grid_c48, ls_mask])

    # open verification
    logger.info("Opening verification data")
    verification_c48 = load_verification(verification_entries, catalog)

    # open prognostic run data
    path = os.path.join(url, "atmos_dt_atmos.zarr")
    logger.info(f"Opening prognostic run data at {path}")
    ds = _load_standardized(path)
    input_grid, coarsening_factor = _get_coarsening_args(ds, 48)
    area = catalog[input_grid].to_dask()["area"]
    ds = _coarsen(ds, area, coarsening_factor)

    return ds, verification_c48, grid_c48


def load_physics(
    url: str, verification_entries: Sequence[str], catalog: intake.Catalog
) -> DiagArg:
    """Open data required for physics plots.

    Args:
        url: path to prognostic run directory
        verification_entries: catalog entries for verification physics data
        catalog: Intake catalog of available data sources

    Returns:
        tuple of prognostic run data, verification data and grid variables all at
        coarsened resolution. Prognostic and verification data contain variables
        output by the physics routines.
    """
    logger.info(f"Processing physics data from run directory at {url}")

    # open grid
    logger.info("Opening Grid Spec")
    grid_c48 = standardize_fv3_diagnostics(catalog["grid/c48"].to_dask())
    ls_mask = standardize_fv3_diagnostics(catalog["landseamask/c48"].to_dask())
    grid_c48 = xr.merge([grid_c48, ls_mask])

    # open verification
    verification_c48 = load_verification(verification_entries, catalog)
    verification_c48 = add_derived.physics_variables(verification_c48)

    # open prognostic run data
    logger.info(f"Opening prognostic run data at {url}")
    prognostic_output = _load_prognostic_run_physics_output(url)
    input_grid, coarsening_factor = _get_coarsening_args(prognostic_output, 48)
    area = catalog[input_grid].to_dask()["area"]
    prognostic_output = _coarsen(prognostic_output, area, coarsening_factor)
    prognostic_output = add_derived.physics_variables(prognostic_output)

    return prognostic_output, verification_c48, grid_c48
