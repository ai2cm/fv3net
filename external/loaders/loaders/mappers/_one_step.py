import os
from typing import Mapping, Sequence
import xarray as xr

import vcm
from vcm import cloud, safe
from ._base import GeoMapper
from .._utils import assign_net_physics_terms

from ..constants import DERIVATION_DIM, DERIVATION_SHIELD_COORD, DERIVATION_FV3GFS_COORD

TIME_DIM_NAME = "initial_time"
DIMENSION_ORDER = ("tile", "z", "y", "y_interface", "x", "x_interface")
SHIELD_SUFFIX = "prog"
ONE_STEP_SUFFIX = "train"


RENAME_ONE_STEP_SHIELD_DIAG_VARS = {
    "total_sky_downward_shortwave_flux_at_top_of_atmosphere": (
        "DSWRFtoa_prog",
        "DSWRFtoa_train",
    ),
    "total_sky_downward_shortwave_flux_at_surface": ("DSWRFsfc_prog", "DSWRFsfc_train"),
    "total_sky_upward_shortwave_flux_at_top_of_atmosphere": (
        "USWRFtoa_prog",
        "USWRFtoa_train",
    ),
    "total_sky_upward_shortwave_flux_at_surface": ("USWRFsfc_prog", "USWRFsfc_train"),
    "total_sky_downward_longwave_flux_at_surface": ("DLWRFsfc_prog", "DLWRFsfc_train"),
    "total_sky_upward_longwave_flux_at_top_of_atmosphere": (
        "ULWRFtoa_prog",
        "ULWRFtoa_train",
    ),
    "total_sky_upward_longwave_flux_at_surface": ("ULWRFsfc_prog", "ULWRFsfc_train"),
    "sensible_heat_flux": ("sensible_heat_flux_prog", "sensible_heat_flux"),
    "latent_heat_flux": ("latent_heat_flux_prog", "latent_heat_flux"),
    "surface_precipitation_rate": (
        "surface_precipitation_rate_prog",
        "surface_precipitation_rate",
    ),
}


class TimestepMapper(GeoMapper):
    def __init__(
        self,
        timesteps_dir: str,
        rename_vars: Mapping[str, str] = None,
        drop_vars: Sequence[str] = None,
        dim_order: Sequence[str] = None,
    ):
        self._timesteps_dir = timesteps_dir
        self._fs = cloud.get_fs(timesteps_dir)
        self._rename_vars = rename_vars or {}
        self._drop_vars = drop_vars
        self._dim_order = dim_order
        self.zarrs = self._fs.glob(os.path.join(timesteps_dir, "*.zarr"))
        if len(self.zarrs) == 0:
            raise ValueError(f"No zarrs found in {timesteps_dir}")

    def __getitem__(self, key: str) -> xr.Dataset:
        zarr_path = os.path.join(self._timesteps_dir, f"{key}.zarr")
        ds = (
            xr.open_zarr(self._fs.get_mapper(zarr_path))
            .squeeze()
            .rename(self._rename_vars)
            .drop_vars(names=self._drop_vars)
            .transpose(*self._dim_order)
        )
        return ds

    def keys(self):
        return set([vcm.parse_timestep_str_from_path(zarr) for zarr in self.zarrs])


class TimestepMapperWithDiags(GeoMapper):
    def __init__(self, timestep_mapper: Mapping[str, xr.Dataset]):
        self._timestep_mapper = timestep_mapper

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = self._timestep_mapper[key]
        ds = self._reshape_one_step_diags(ds)
        return assign_net_physics_terms(ds)

    def keys(self):
        return self._timestep_mapper.keys()

    @staticmethod
    def _reshape_one_step_diags(
        ds: xr.Dataset,
        reshape_vars: Mapping[str, str] = RENAME_ONE_STEP_SHIELD_DIAG_VARS,
        shield_suffix: str = SHIELD_SUFFIX,
        one_step_suffix: str = ONE_STEP_SUFFIX,
        overlap_dim: str = DERIVATION_DIM,
    ) -> xr.Dataset:

        overlap_dim_vars = {}
        for rename, reshape_vars in reshape_vars.items():
            var_da = (
                safe.get_variables(ds, reshape_vars)
                .to_array(dim=overlap_dim)
                .assign_coords(
                    {overlap_dim: [DERIVATION_SHIELD_COORD, DERIVATION_FV3GFS_COORD]}
                )
            )
            overlap_dim_vars[rename] = var_da
            ds = ds.drop_vars(names=reshape_vars)

        return ds.assign(overlap_dim_vars)


def open_one_step(
    url: str,
    add_shield_diags: bool = False,
    rename_vars: Mapping[str, str] = None,
    drop_vars: Sequence[str] = (TIME_DIM_NAME,),
    dim_order: Sequence[str] = DIMENSION_ORDER,
) -> Mapping[str, xr.Dataset]:

    if not add_shield_diags:
        mapper = TimestepMapper(url, rename_vars, drop_vars, dim_order)
    else:
        mapper = TimestepMapperWithDiags(
            TimestepMapper(url, rename_vars, drop_vars, dim_order)
        )
    return mapper
