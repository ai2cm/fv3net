from typing import (
    Hashable,
    MutableMapping,
)
import xarray as xr

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]
Tendencies = MutableMapping[Hashable, xr.DataArray]
