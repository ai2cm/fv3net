import os
from typing import Mapping, Sequence
import xarray as xr

import vcm
from vcm import cloud
from ._base import GeoMapper

TIME_DIM_NAME = "initial_time"
DIMENSION_ORDER = ("tile", "z", "y", "y_interface", "x", "x_interface")


def open_one_step(
    url: str,
    rename_vars: Mapping[str, str] = None,
    drop_vars: Sequence[str] = (TIME_DIM_NAME,),
    dim_order: Sequence[str] = DIMENSION_ORDER,
) -> Mapping[str, xr.Dataset]:
    return TimestepMapper(url, rename_vars, drop_vars, dim_order)


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
