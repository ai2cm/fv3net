import numpy as np
from typing import Mapping, Hashable, Callable, Union, List

import xarray as xr
import vcm
from ._base import DerivedState


# TODO: this class is not yet complete.
# For the initial PR just want to move DerivedState out of prognostic runtime.
# Subsequent PR will change the batches and offline diags code to use this class
# and fill out the missing functionality.


class DerivedDatasetState(DerivedState):
    _VARIABLES: Mapping[Hashable, Callable[..., xr.DataArray]] = {}

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def __getitem__(self, key: Hashable) -> Union[xr.Dataset, xr.DataArray]:
        if len(key) > 1:
            return self._get_dataset(key)
        elif key in self._VARIABLES:
            return self._VARIABLES[key](self)
        else:
            return self.ds[key]

    def _get_dataset(self, keys: List[str]):
        derived = [var for var in keys if var in self._VARIABLES]
        nonderived = [var for var in keys if var not in self._VARIABLES]

        ds_nonderived = vcm.safe.get_variables(self.ds, nonderived)
        ds_derived = xr.Dataset({var: self._VARIABLES[var](self) for var in derived})
        return xr.merge([ds_nonderived, ds_derived])


@DerivedDatasetState.register("cos_zenith_angle")
def cos_zenith_angle(self):
    times_exploded = np.array(
        [
            np.full(self["lon"].shape, vcm.cast_to_datetime(t))
            for t in self["time"].values
        ]
    )
    cos_z = vcm.cos_zenith_angle(times_exploded, self["lon"], self["lat"])
    cos_z_dims = ("time",) + self["lon"].dims
    return xr.DataArray(cos_z, dims=cos_z_dims)


@DerivedDatasetState.register("eastward_wind_tendency")
def eastward_wind_tendency(self):
    pass


@DerivedDatasetState.register("northward_wind_tendency")
def northward_wind_tendency(self):
    pass
