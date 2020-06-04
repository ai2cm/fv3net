import os
from typing import Mapping
import xarray as xr

import vcm
from vcm import cloud


def open_one_step(url: str) -> Mapping[str, xr.Dataset]:
    return TimestepMapper(url)


class TimestepMapper:
    def __init__(self, timesteps_dir):
        self._timesteps_dir = timesteps_dir
        self._fs = cloud.get_fs(timesteps_dir)
        self.zarrs = self._fs.glob(os.path.join(timesteps_dir, "*.zarr"))
        if len(self.zarrs) == 0:
            raise ValueError(f"No zarrs found in {timesteps_dir}")

    def __getitem__(self, key: str) -> xr.Dataset:
        zarr_path = os.path.join(self._timesteps_dir, f"{key}.zarr")
        return xr.open_zarr(self._fs.get_mapper(zarr_path))

    def keys(self):
        return [vcm.parse_timestep_str_from_path(zarr) for zarr in self.zarrs]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())
