from typing import (
    Hashable,
    Callable,
    MutableMapping,
)
import xarray as xr

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]
Tendencies = MutableMapping[Hashable, xr.DataArray]
Step = Callable[[], Diagnostics]
