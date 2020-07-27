import os
import re
import vcm
from vcm import parse_datetime_from_str, safe
from typing import Mapping, Optional, Union, Sequence, Tuple, Union
import numpy as np
from functools import partial

import xarray as xr
from toolz import groupby

from ._transformations import KeyMap
from .._utils import assign_net_physics_terms
from ..constants import (
    DERIVATION_FV3GFS_COORD,
    DERIVATION_SHIELD_COORD,
    RENAMED_SHIELD_DIAG_VARS,
)
from ._base import GeoMapper
from ._high_res_diags import open_high_res_diags
from ._merged import MergeOverlappingData

Time = str
Tile = int
K = Tuple[Time, Tile]

# TODO: Ensure units and long_name metadata are updated
NEW_NAMES = {
    "T": "air_temperature",
    "t_dt_fv_sat_adj_coarse": "air_temperature_saturation_adjustment",
    "t_dt_nudge_coarse": "air_temperature_nudging",
    "t_dt_phys_coarse": "air_temperature_physics",
    "eddy_flux_vulcan_omega_temp": "air_temperature_unresolved_flux",
    "T_vulcan_omega_coarse": "air_temperature_total_resolved_flux",
    "T_storage": "air_temperature_storage",
    "sphum": "specific_humidity",
    "qv_dt_fv_sat_adj_coarse": "specific_humidity_saturation_adjustment",
    "qv_dt_phys_coarse": "specific_humidity_physics",
    "eddy_flux_vulcan_omega_sphum": "specific_humidity_unresolved_flux",
    "sphum": "specific_humidity",
    "sphum_vulcan_omega_coarse": "specific_humidity_total_resolved_flux",
    "sphum_storage": "specific_humidity_storage",
    "vulcan_omega_coarse": "omega",
}


def eddy_flux_coarse(unresolved_flux, total_resolved_flux, omega, field):
    """Compute re-coarsened eddy flux divergence from re-coarsed data
    """
    return unresolved_flux + (total_resolved_flux - omega * field)


def _center_to_interface(f: np.ndarray) -> np.ndarray:
    """Interpolate vertically cell centered data to the interface
    with linearly extrapolated inputs"""
    f_low = 2 * f[..., 0] - f[..., 1]
    f_high = 2 * f[..., -1] - f[..., -2]
    pad = np.concatenate([f_low[..., np.newaxis], f, f_high[..., np.newaxis]], axis=-1)
    return (pad[..., :-1] + pad[..., 1:]) / 2


def _convergence(eddy: np.ndarray, delp: np.ndarray) -> np.ndarray:
    """Compute vertical convergence of a cell-centered flux.

    This flux is assumed to vanish at the vertical boundaries
    """
    padded = _center_to_interface(eddy)
    # pad interfaces assuming eddy = 0 at edges
    return -np.diff(padded, axis=-1) / delp


def convergence(eddy: xr.DataArray, delp: xr.DataArray, dim: str = "p") -> xr.DataArray:
    return xr.apply_ufunc(
        _convergence,
        eddy,
        delp,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[eddy.dtype],
    )


class FineResolutionBudgetTiles(GeoMapper):
    """An Mapping interface to a fine-res-q1-q2 dataset"""

    def __init__(self, url):
        self._fs = vcm.cloud.get_fs(url)
        self._url = url
        self.files = self._fs.glob(os.path.join(url, "*.nc"))
        if len(self.files) == 0:
            raise ValueError("No file detected")

    def _parse_file(self, url):
        pattern = r"tile(.)\.nc"
        match = re.search(pattern, url)
        date = vcm.parse_timestep_str_from_path(url)
        tile = match.group(1)
        return date, int(tile)

    def __getitem__(self, key: str) -> xr.Dataset:
        return vcm.open_remote_nc(self._fs, self._find_file(key))

    def _find_file(self, key):
        return [file for file in self.files if self._parse_file(file) == key][-1]

    def keys(self):
        return [self._parse_file(file) for file in self.files]


class GroupByTime(GeoMapper):
    def __init__(self, tiles: Mapping[K, xr.Dataset]) -> Mapping[K, xr.Dataset]:
        def fn(key):
            time, _ = key
            return time

        self._tiles = tiles
        self._time_lookup = groupby(fn, self._tiles.keys())

    def keys(self):
        return self._time_lookup.keys()

    def __getitem__(self, time: Time) -> xr.Dataset:
        datasets = [self._tiles[key] for key in self._time_lookup[time]]
        tiles = range(len(datasets))
        return xr.concat(datasets, dim="tile").assign_coords(tile=tiles)


class FineResolutionSources(GeoMapper):
    def __init__(
        self,
        fine_resolution_time_mapping: Mapping[Time, xr.Dataset],
        offset_seconds: Union[int, float] = 0,
        drop_vars: Sequence[str] = ("step", "time"),
        dim_order: Sequence[str] = ("tile", "z", "y", "x"),
        rename_vars: Optional[Mapping[str, str]] = None,
    ):
        self._time_mapping = fine_resolution_time_mapping
        self._offset_seconds = offset_seconds
        self._rename_vars = rename_vars or {}
        print(self._rename_vars)
        self._drop_vars = drop_vars
        self._dim_order = dim_order
        self._rename_vars = rename_vars or {}

    def keys(self):
        return self._time_mapping.keys()

    def __getitem__(self, time: Time) -> xr.Dataset:
        return (
            self._derived_budget_ds(self._time_mapping[time])
            .drop_vars(names=self._drop_vars, errors="ignore")
            .rename(self._rename_vars)
            .transpose(*self._dim_order)
        )

    def _rename_budget_inputs_ds(self, budget_time_ds: xr.Dataset) -> xr.Dataset:
        return budget_time_ds.rename(NEW_NAMES)

    def _compute_coarse_eddy_flux_convergence_ds(
        self, budget_time_ds: xr.Dataset, field: str, vertical_dimension: str
    ) -> xr.Dataset:
        eddy_flux = eddy_flux_coarse(
            budget_time_ds[f"{field}_unresolved_flux"],
            budget_time_ds[f"{field}_total_resolved_flux"],
            budget_time_ds["omega"],
            budget_time_ds[field],
        )
        budget_time_ds[f"{field}_convergence"] = convergence(
            eddy_flux, budget_time_ds["delp"], dim=vertical_dimension
        )
        return budget_time_ds

    def _derived_budget_ds(
        self,
        budget_time_ds: xr.Dataset,
        variable_prefixes: Mapping[str, str] = None,
        apparent_source_terms: Sequence[str] = (
            "physics",
            "saturation_adjustment",
            "convergence",
        ),
        vertical_dimension: str = "pfull",
    ) -> xr.Dataset:

        if variable_prefixes is None:
            variable_prefixes = {
                "air_temperature": "Q1",
                "specific_humidity": "Q2",
            }

        budget_time_ds = self._rename_budget_inputs_ds(budget_time_ds)
        for variable_name, apparent_source_name in variable_prefixes.items():
            budget_time_ds = (
                budget_time_ds.pipe(
                    self._compute_coarse_eddy_flux_convergence_ds,
                    variable_name,
                    vertical_dimension,
                )
                .pipe(
                    self._insert_budget_dQ,
                    variable_name,
                    f"d{apparent_source_name}",
                    apparent_source_terms,
                )
                .pipe(
                    self._insert_budget_pQ, variable_name, f"p{apparent_source_name}",
                )
            )
        budget_time_ds = budget_time_ds.pipe(self._insert_physics).pipe(
            assign_net_physics_terms
        )
        return budget_time_ds

    @staticmethod
    def _insert_budget_dQ(
        budget_time_ds: xr.Dataset,
        variable_name: str,
        apparent_source_name: str,
        apparent_source_terms: Sequence[str],
    ) -> xr.Dataset:
        """Insert dQ (really Q) from other budget terms"""

        source_vars = [f"{variable_name}_{term}" for term in apparent_source_terms]
        apparent_source = (
            vcm.safe.get_variables(budget_time_ds, source_vars)
            .to_array(dim="variable")
            .sum(dim="variable")
        )
        budget_time_ds = budget_time_ds.assign({apparent_source_name: apparent_source})

        units = budget_time_ds[f"{variable_name}_{apparent_source_terms[0]}"].attrs.get(
            "units", None
        )
        budget_time_ds[apparent_source_name].attrs.update(
            {"name": f"apparent source of {variable_name}"}
        )
        if units is not None:
            budget_time_ds[apparent_source_name].attrs.update({"units": units})

        return budget_time_ds

    @staticmethod
    def _insert_budget_pQ(
        budget_time_ds: xr.Dataset, variable_name: str, apparent_source_name: str,
    ) -> xr.Dataset:
        """Insert pQ = 0 in the fine-res budget case"""

        budget_time_ds = budget_time_ds.assign(
            {apparent_source_name: xr.zeros_like(budget_time_ds[f"{variable_name}"])}
        )

        budget_time_ds[apparent_source_name].attrs[
            "name"
        ] = f"coarse-res physics tendency of {variable_name}"

        units = budget_time_ds[f"{variable_name}"].attrs.get("units", None)
        if units is not None:
            budget_time_ds[apparent_source_name].attrs["units"] = f"{units}/s"

        return budget_time_ds

    @staticmethod
    def _insert_physics(
        budget_time_ds: xr.Dataset,
        physics_varnames: Sequence[str] = RENAMED_SHIELD_DIAG_VARS.values(),
    ) -> xr.Dataset:

        template_2d_var = budget_time_ds["air_temperature"].isel({"pfull": 0})

        physics_vars = {}
        for var in physics_varnames:
            physics_var = xr.full_like(template_2d_var, fill_value=0.0)
            physics_vars[var] = physics_var

        return budget_time_ds.assign(physics_vars)


def open_fine_resolution_budget(url: str) -> Mapping[str, xr.Dataset]:
    """Open a mapping interface to the fine resolution budget data

    Example:

        >>> from loaders import open_fine_resolution_budget
        >>> loader = open_fine_resolution_budget('gs://vcm-ml-scratch/noah/2020-05-19/')
        >>> len(loader)
        479
        >>> loader['20160805.202230']
        <xarray.Dataset>
        Dimensions:                         (grid_xt: 48, grid_yt: 48, pfull: 79, tile: 6)
        Coordinates:
            time                            object 2016-08-05 20:22:30
            step                            <U6 'middle'
        * tile                            (tile) int64 1 2 3 4 5 6
        Dimensions without coordinates: grid_xt, grid_yt, pfull
        Data variables:
            air_temperature                 (tile, pfull, grid_yt, grid_xt) float32 235.28934 ... 290.56107
            air_temperature_convergence     (tile, grid_yt, grid_xt, pfull) float32 4.3996937e-07 ... 1.7985441e-06
            air_temperature_eddy            (tile, pfull, grid_yt, grid_xt) float32 -2.3193044e-05 ... 0.0004279223
            air_temperature_microphysics    (tile, pfull, grid_yt, grid_xt) float32 0.0 ... -5.5472506e-06
            air_temperature_nudging         (tile, pfull, grid_yt, grid_xt) float32 0.0 ... 2.0156076e-06
            air_temperature_physics         (tile, pfull, grid_yt, grid_xt) float32 2.3518855e-06 ... -3.3252392e-05
            air_temperature_resolved        (tile, pfull, grid_yt, grid_xt) float32 0.26079428 ... 0.6763954
            air_temperature_storage         (tile, pfull, grid_yt, grid_xt) float32 0.000119928314 ... 5.2825694e-06
            specific_humidity               (tile, pfull, grid_yt, grid_xt) float32 5.7787e-06 ... 0.008809893
            specific_humidity_convergence   (tile, grid_yt, grid_xt, pfull) float32 -6.838638e-14 ... -1.7079346e-08
            specific_humidity_eddy          (tile, pfull, grid_yt, grid_xt) float32 -1.0437861e-13 ... -2.5796332e-06
            specific_humidity_microphysics  (tile, pfull, grid_yt, grid_xt) float32 0.0 ... 1.6763515e-09
            specific_humidity_physics       (tile, pfull, grid_yt, grid_xt) float32 -1.961625e-14 ... 5.385441e-09
            specific_humidity_resolved      (tile, pfull, grid_yt, grid_xt) float32 6.4418755e-09 ... 2.0072384e-05
            specific_humidity_storage       (tile, pfull, grid_yt, grid_xt) float32 -6.422655e-11 ... -5.3609618e-08
            Example:
    """  # noqa
    tiles = FineResolutionBudgetTiles(url)
    return GroupByTime(tiles)


def open_fine_res_apparent_sources(
    fine_res_url: str,
    shield_diags_url: str = None,
    offset_seconds: Union[int, float] = 0,
    rename_vars: Mapping[str, str] = None,
    dim_order: Sequence[str] = ("tile", "z", "y", "x"),
    drop_vars: Sequence[str] = ("step", "time"),
) -> Mapping[str, xr.Dataset]:
    """Open a derived mapping interface to the fine resolution budget, grouped
        by time and with derived apparent sources

    Args:
        fine_res_url (str): path to fine res dataset
        shield_diags_url: path to directory containing a zarr store of SHiELD
            diagnostics coarsened to the nudged model resolution (optional)
        offset_seconds: amount to shift the keys forward by in seconds. For
            example, if the underlying data contains a value at the key
            "20160801.000730", a value off 450 will shift this forward 7:30
            minutes, so that this same value can be accessed with the key
            "20160801.001500"
        rename_vars: (mapping): optional mapping of variables to rename in dataset
        drop_vars (sequence): optional list of variable names to drop from dataset
    """

    # use default which is valid for real data
    if rename_vars is None:
        rename_vars = {"grid_xt": "x", "grid_yt": "y", "pfull": "z"}

    fine_resolution_sources_mapper = FineResolutionSources(
        open_fine_resolution_budget(fine_res_url),
        drop_vars,
        dim_order=dim_order,
        rename_vars=rename_vars,
    )

    fine_resolution_sources_mapper = KeyMap(
        partial(vcm.shift_timestamp, seconds=offset_seconds),
        fine_resolution_sources_mapper,
    )

    if shield_diags_url is not None:
        shield_diags_mapper = open_high_res_diags(shield_diags_url)
        fine_resolution_sources_mapper = MergeOverlappingData(
            shield_diags_mapper,
            fine_resolution_sources_mapper,
            source_name_left=DERIVATION_SHIELD_COORD,
            source_name_right=DERIVATION_FV3GFS_COORD,
        )

    return fine_resolution_sources_mapper
