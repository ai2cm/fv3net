from dataclasses import dataclass
import json
import logging
import os
from typing_extensions import Protocol
from typing import List, Mapping
import warnings

import fsspec
import intake
import pandas as pd
import vcm
import xarray as xr
from vcm.cloud import get_fs
from vcm.fv3 import standardize_fv3_diagnostics
from vcm.scream import standardize_scream_diagnostics
from vcm import check_if_scream_dataset

from fv3net.diagnostics.prognostic_run import config
from fv3net.diagnostics.prognostic_run import derived_variables
from fv3net.diagnostics.prognostic_run import constants


logger = logging.getLogger(__name__)


def load_verification(
    catalog_keys: List[str], catalog: intake.catalog.Catalog, join="outer"
) -> xr.Dataset:

    """
    Load verification data sources from a catalog and combine for reporting.

    Args:
        catalog_keys: catalog sources to load as verification data
        catalog: Intake catalog of available data sources.
        join: how to join verification data sources.

    Returns:
        All specified verification datasources standardized and merged

    """
    verif_data = []
    for dataset_key in catalog_keys:
        ds = catalog[dataset_key].to_dask()
        ds = standardize_fv3_diagnostics(ds)
        verif_data.append(ds)
    return xr.merge(verif_data, join=join)


def _load_standardized(path):
    logger.info(f"Loading and standardizing {path}")
    m = fsspec.get_mapper(path)
    ds = xr.open_zarr(m, consolidated=True, decode_times=False)
    return standardize_fv3_diagnostics(ds)


def _get_area(ds: xr.Dataset, catalog: intake.catalog.Catalog) -> xr.DataArray:
    grid_entries = {
        48: "grid/c48",
        96: "grid/c96",
        384: "grid/c384",
        21600: "grid/ne30",
    }
    if check_if_scream_dataset(ds):
        input_res = ds.sizes["ncol"]
    else:
        input_res = ds.sizes["x"]
    if input_res not in grid_entries:
        raise KeyError(f"No grid defined in catalog for c{input_res} resolution")
    return catalog[grid_entries[input_res]].to_dask().area


def _get_factor(ds: xr.Dataset, target_resolution: int) -> int:
    if check_if_scream_dataset(ds):
        input_res = ds.sizes["ncol"]
    else:
        input_res = ds.sizes["x"]
    if input_res % target_resolution != 0:
        raise ValueError("Target resolution must evenly divide input resolution")
    return int(input_res / target_resolution)


def _coarsen_cell_centered_to_target_resolution(
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
    files_3d = [
        "diags_3d.zarr",
        "state_after_timestep.zarr",
        "nudging_tendencies.zarr",
        "piggy.zarr",
    ]
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


def load_grid(catalog, gsrm="fv3gfs"):
    logger.info("Opening Grid Spec")
    if gsrm == "fv3gfs":
        grid_c48 = standardize_fv3_diagnostics(catalog["grid/c48"].to_dask())
        ls_mask = standardize_fv3_diagnostics(catalog["landseamask/c48"].to_dask())
        return xr.merge([grid_c48, ls_mask])
    elif gsrm == "scream":
        grid_ne30 = standardize_scream_diagnostics(catalog["grid/ne30"].to_dask())
        ls_mask_ne30 = standardize_scream_diagnostics(
            catalog["landseamask/ne30"].to_dask()
        )
        return xr.merge([grid_ne30, ls_mask_ne30])
    else:
        raise ValueError(f"Grid spec {gsrm} not supported")


def load_coarse_data(path, catalog) -> xr.Dataset:
    logger.info(f"Opening prognostic run data at {path}")

    try:
        ds = _load_standardized(path)
    except (FileNotFoundError, KeyError):
        warnings.warn(UserWarning(f"{path} not found. Returning empty dataset."))
        ds = xr.Dataset()

    if len(ds) > 0:
        # drop interface vars to avoid broadcasting by coarsen func
        ds = ds.drop_vars(
            constants.GRID_VARS
            + constants.GRID_INTERFACE_COORDS
            + constants.FORTRAN_TILE_ONLY_VARS,
            errors="ignore",
        )
        ds = _coarsen_cell_centered_to_target_resolution(
            ds, target_resolution=48, catalog=catalog
        )

    return ds


def _get_physics_only_contribution(ds):
    
    _tendencies_to_separate = {
        "dQ1": "air_temperature",
        "dQ2": "specific_humidity",
        "dQu": "eastward_wind",
        "dQv": "northward_wind",
        "air_temperature_tendency_due_to_nudging": "air_temperature",
        "specific_humidity_tendency_due_to_nudging": "specific_humidity",
        "eastward_wind_tendency_due_to_nudging": "eastward_wind",
        "northward_wind_tendency_due_to_nudging": "northward_wind",
    }

    adjusted_physics = {}
    for adjustment_key, physics_varname in _tendencies_to_separate.items():
        if adjustment_key in ds:
            logger.info(f"Removing {adjustment_key} tendency from total scream physics tendency")
            adjustment_tend = ds[adjustment_key]
            physics_key = f"{physics_varname}_tendency_due_to_scream_physics"
            total_physics = ds[physics_key]
            adjusted_physics[physics_key] = total_physics - adjustment_tend
    
    return ds.update(adjusted_physics)


def _fix_moistening_and_prate_units_for_diags(ds):

    prate_key = "PRATEsfc"
    dq2_key = "net_moistening_due_to_machine_learning"
    if prate_key in ds and dq2_key in ds:
        if ds[prate_key].units == "kg/m^2/s" and ds[dq2_key].units == "kg/m^2/s":
            logger.info("Removing ML precip from SCREAM PRATEsfc")
            ds[prate_key] = ds[prate_key] - ds[dq2_key]
        else:
            logger.warning(
                "PRATEsfc and net_moistening_due_to_machine_learning "
                "units do not match, could not remove ML precip from PRATEsfc. "
                "physics heating and moistening diagnostics may be incorrect."
            )

    if dq2_key in ds:
        logger.info("Changing units of ml net moistening to mm/day")
        ds[dq2_key] = ds[dq2_key] * derived_variables.SECONDS_PER_DAY
    
    return ds


def load_scream_data(path) -> xr.Dataset:
    logger.info(f"Opening prognostic run data at {path}")

    try:
        logger.info(f"Loading and standardizing {path}")
        m = fsspec.get_mapper(path)
        ds = xr.open_zarr(m, consolidated=True, decode_times=False)
        ds = standardize_scream_diagnostics(ds)
        ds = _get_physics_only_contribution(ds)
        ds = _fix_moistening_and_prate_units_for_diags(ds)
    except (FileNotFoundError):
        warnings.warn(UserWarning(f"{path} not found. Returning empty dataset."))
        ds = xr.Dataset()
    return ds


def loads_stats(b: bytes):
    lines = b.decode().splitlines(keepends=False)
    return [json.loads(line) for line in lines]


def open_segmented_stats(url: str) -> pd.DataFrame:
    fs = get_fs(url)
    logfiles = sorted(fs.glob(f"{url}/**/statistics.txt"))
    records: List[Mapping] = sum(
        [loads_stats(fs.cat(logfile)) for logfile in logfiles], []
    )
    return pd.DataFrame.from_records(records)


def open_segmented_logs(url: str) -> vcm.fv3.logs.FV3Log:
    fs = get_fs(url)
    logfiles = sorted(fs.glob(f"{url}/**/logs.txt"))
    logs = [vcm.fv3.logs.loads(fs.cat(url).decode()) for url in logfiles]
    return vcm.fv3.logs.concatenate(logs)


def open_segmented_logs_as_strings(url: str) -> List[str]:
    """Open the logs from each segment of a segmented run as strings
    """
    fs = vcm.get_fs(url)
    logfiles = sorted(fs.glob(f"{url}/**/logs.txt"))
    logs = [fs.cat(url).decode() for url in logfiles]
    return logs


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
    join_2d: str = "outer"

    @property
    def _verif_entries(self):
        return config.get_verification_entries(self.tag, self.catalog)

    @property
    def _rename_map(self):
        return constants.VERIFICATION_RENAME_MAP.get(self.tag, {})

    @property
    def data_2d(self) -> xr.Dataset:
        return load_verification(
            self._verif_entries["2d"], self.catalog, join=self.join_2d
        ).rename(self._rename_map.get("2d", {}))

    @property
    def data_3d(self) -> xr.Dataset:
        return load_verification(self._verif_entries["3d"], self.catalog).rename(
            self._rename_map.get("3d", {})
        )

    def __str__(self) -> str:
        return self.tag


@dataclass
class SegmentedRun:
    url: str
    catalog: intake.catalog.base.Catalog
    join_2d: str = "outer"

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
            join=self.join_2d,
        )

    @property
    def data_3d(self) -> xr.Dataset:
        return _load_3d(self.url, self.catalog)

    @property
    def artifacts(self) -> List[str]:
        url = self.url
        fs = vcm.get_fs(url)
        # to ensure up to date results
        fs.invalidate_cache()
        return sorted(fs.ls(f"{url}/artifacts"))

    def __str__(self) -> str:
        return self.url


@dataclass
class ScreamSimulation:
    url: str

    @property
    def data_2d(self) -> xr.Dataset:
        url = self.url
        path = os.path.join(url, "data_2d.zarr")

        return load_scream_data(path)

    @property
    def data_3d(self) -> xr.Dataset:
        logger.info(f"Processing 3d data from run directory at {self.url}")
        path = os.path.join(self.url, "data_3d.zarr")
        ds = load_scream_data(path)

        # interpolate 3d prognostic fields to pressure levels
        ds_interp = xr.Dataset()
        pressure_vars = [var for var in ds.data_vars if "z" in ds[var].dims]
        for var in pressure_vars:
            ds_interp[var] = vcm.interpolate_to_pressure_levels(
                field=ds[var],
                delp=ds["pressure_thickness_of_atmospheric_layer"],
                dim="z",
            )

        return ds_interp

    def __str__(self) -> str:
        return self.url


def evaluation_pair_to_input_data(
    prognostic: Simulation,
    verification: Simulation,
    grid: xr.Dataset,
    start_date: str = None,
    end_date: str = None,
):
    # 3d data special handling
    data_3d = prognostic.data_3d
    verif_3d = verification.data_3d
    data_2d = prognostic.data_2d
    verif_2d = verification.data_2d
    if check_if_scream_dataset(data_3d):
        dropped_grid_vars = ["land_sea_mask"]
    else:
        dropped_grid_vars = ["tile", "land_sea_mask"]
    data_3d = data_3d.sel(time=slice(start_date, end_date))
    verif_3d = verif_3d.sel(time=slice(start_date, end_date))
    data_2d = data_2d.sel(time=slice(start_date, end_date))
    verif_2d = verif_2d.sel(time=slice(start_date, end_date))
    return {
        "3d": (
            derived_variables.derive_3d_variables(data_3d),
            derived_variables.derive_3d_variables(verif_3d),
            grid.drop(dropped_grid_vars),
        ),
        "2d": (
            derived_variables.derive_2d_variables(data_2d),
            derived_variables.derive_2d_variables(verif_2d),
            grid,
        ),
    }
