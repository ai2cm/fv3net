from dataclasses import dataclass
import json
import logging
import os
from typing_extensions import Protocol
from typing import List
import warnings

import fsspec
import intake
import pandas as pd
import vcm
import xarray as xr
from vcm.cloud import get_fs
from vcm.fv3 import standardize_fv3_diagnostics

from fv3net.diagnostics.prognostic_run import config
from fv3net.diagnostics.prognostic_run import derived_variables

logger = logging.getLogger(__name__)

GRID_VARS = ["lat", "lon", "latb", "lonb"]
DROP_DIMS = ["x_interface", "y_interface"]


def load_verification(
    catalog_keys: List[str], catalog: intake.catalog.Catalog,
) -> xr.Dataset:

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


def _get_area(ds: xr.Dataset, catalog: intake.catalog.Catalog) -> xr.DataArray:
    grid_entries = {48: "grid/c48", 96: "grid/c96", 384: "grid/c384"}
    input_res = ds.sizes["x"]
    if input_res not in grid_entries:
        raise KeyError(f"No grid defined in catalog for c{input_res} resolution")
    return catalog[grid_entries[input_res]].to_dask().area


def _get_factor(ds: xr.Dataset, target_resolution: int) -> int:
    input_res = ds.sizes["x"]
    if input_res % target_resolution != 0:
        raise ValueError("Target resolution must evenly divide input resolution")
    return int(input_res / target_resolution)


def _coarsen_to_target_resolution(
    ds: xr.Dataset, target_resolution: int, catalog: intake.catalog.Catalog,
) -> xr.Dataset:
    return vcm.cubedsphere.weighted_block_average(
        ds,
        weights=_get_area(ds, catalog),
        coarsening_factor=_get_factor(ds, target_resolution),
        x_dim="x",
        y_dim="y",
    )


def _load_3d(url: str, catalog: intake.catalog.Catalog) -> xr.Dataset:
    logger.info(f"Processing 3d data from run directory at {url}")
    files_3d = ["diags_3d.zarr", "state_after_timestep.zarr", "nudging_tendencies.zarr"]
    ds = xr.merge(
        [
            load_coarse_data(os.path.join(url, filename), catalog)
            for filename in files_3d
        ]
    )

    # interpolate 3d prognostic fields to pressure levels
    ds_interp = xr.Dataset()
    pressure_vars = [var for var in ds.data_vars if "z" in ds[var].dims]
    for var in pressure_vars:
        ds_interp[var] = vcm.interpolate_to_pressure_levels(
            field=ds[var], delp=ds["pressure_thickness_of_atmospheric_layer"], dim="z",
        )

    return ds_interp


def load_grid(catalog):
    logger.info("Opening Grid Spec")
    grid_c48 = standardize_fv3_diagnostics(catalog["grid/c48"].to_dask())
    ls_mask = standardize_fv3_diagnostics(catalog["landseamask/c48"].to_dask())
    return xr.merge([grid_c48, ls_mask])


def load_coarse_data(path, catalog) -> xr.Dataset:
    logger.info(f"Opening prognostic run data at {path}")

    try:
        ds = _load_standardized(path)
    except (FileNotFoundError, KeyError):
        warnings.warn(UserWarning(f"{path} not found. Returning empty dataset."))
        ds = xr.Dataset()

    if len(ds) > 0:
        ds = ds.drop_vars(GRID_VARS + DROP_DIMS, errors="ignore")
        ds = _coarsen_to_target_resolution(ds, target_resolution=48, catalog=catalog)

    return ds


def loads_stats(b: bytes):
    lines = b.decode().splitlines(keepends=False)
    return [json.loads(line) for line in lines]


def open_segmented_stats(url: str) -> pd.DataFrame:
    fs = get_fs(url)
    logfiles = sorted(fs.glob(f"{url}/**/statistics.txt"))
    records = sum([loads_stats(fs.cat(logfile)) for logfile in logfiles], [])
    return pd.DataFrame.from_records(records)


def open_segmented_logs(url: str) -> vcm.fv3.logs.FV3Log:
    fs = get_fs(url)
    logfiles = sorted(fs.glob(f"{url}/**/logs.txt"))
    logs = [vcm.fv3.logs.loads(fs.cat(url).decode()) for url in logfiles]
    return vcm.fv3.logs.concatenate(logs)


class Simulation(Protocol):
    @property
    def data_2d(self) -> xr.Dataset:
        pass

    @property
    def data_3d(self) -> xr.Dataset:
        pass


@dataclass
class CatalogSimulation:
    """A simulation specified in an intake catalog

    Typically used for commonly used runs like the high resolution SHiELD
    simulation, that are specified in a catalog.
    
    """

    tag: str
    catalog: intake.catalog.base.Catalog

    @property
    def _verif_entries(self):
        return config.get_verification_entries(self.tag, self.catalog)

    @property
    def data_2d(self) -> xr.Dataset:
        return load_verification(self._verif_entries["2d"], self.catalog)

    @property
    def data_3d(self) -> xr.Dataset:
        return load_verification(self._verif_entries["3d"], self.catalog)

    def __str__(self) -> str:
        return self.tag


@dataclass
class SegmentedRun:
    url: str
    catalog: intake.catalog.base.Catalog

    @property
    def data_2d(self) -> xr.Dataset:
        url = self.url
        catalog = self.catalog
        path = os.path.join(url, "atmos_dt_atmos.zarr")
        diags_url = os.path.join(url, "diags.zarr")
        sfc_dt_atmos_url = os.path.join(url, "sfc_dt_atmos.zarr")

        return xr.merge(
            [
                load_coarse_data(path, catalog),
                # TODO fillna required because diags.zarr may be saved with an
                # incorrect fill_value. not sure if this is fixed or not.
                load_coarse_data(diags_url, catalog).fillna(0.0),
                load_coarse_data(sfc_dt_atmos_url, catalog),
            ],
            join="outer",
        )

    @property
    def data_3d(self) -> xr.Dataset:
        return _load_3d(self.url, self.catalog)

    def __str__(self) -> str:
        return self.url


def evaluation_pair_to_input_data(
    prognostic: Simulation, verification: Simulation, grid: xr.Dataset
):
    # 3d data special handling
    data_3d = prognostic.data_3d
    verif_3d = verification.data_3d

    return {
        "3d": (data_3d, verif_3d, grid.drop(["tile", "land_sea_mask"]),),
        "2d": (
            derived_variables.physics_variables(prognostic.data_2d),
            derived_variables.physics_variables(verification.data_2d),
            grid,
        ),
    }
