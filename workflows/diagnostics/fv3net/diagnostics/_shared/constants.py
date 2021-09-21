import xarray as xr
from typing import Optional
from dataclasses import dataclass

HORIZONTAL_DIMS = ["x", "y", "tile"]
VERTICAL_DIM = "z"

# argument typehint for diags in save_prognostic_run_diags but used
# by multiple modules split out to group operations and simplify
# the diagnostic script
@dataclass
class DiagArg:
    prediction: xr.Dataset
    verification: xr.Dataset
    grid: xr.Dataset
    delp: Optional[xr.DataArray] = None
