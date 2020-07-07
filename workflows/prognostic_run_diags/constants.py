import xarray as xr
from typing import Tuple

HORIZONTAL_DIMS = ["x", "y", "tile"]

# argument typehint for diags in save_prognostic_run_diags but used
# by multiple modules split out to group operations and simplify
# the diagnostic script
DiagArg = Tuple[xr.Dataset, xr.Dataset, xr.Dataset]
