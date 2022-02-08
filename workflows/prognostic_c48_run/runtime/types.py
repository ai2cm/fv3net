from typing import (
    Hashable,
    Callable,
    MutableMapping,
)
import xarray as xr

import pace.util

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]
Tendencies = MutableMapping[Hashable, xr.DataArray]
Step = Callable[[], Diagnostics]
QuantityState = MutableMapping[Hashable, pace.util.Quantity]
