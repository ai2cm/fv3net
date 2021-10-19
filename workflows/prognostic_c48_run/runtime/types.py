from typing import (
    Hashable,
    Callable,
    MutableMapping,
)
import xarray as xr

import fv3gfs.util

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]
Tendencies = MutableMapping[Hashable, xr.DataArray]
Step = Callable[[], Diagnostics]
QuantityState = MutableMapping[Hashable, fv3gfs.util.Quantity]
