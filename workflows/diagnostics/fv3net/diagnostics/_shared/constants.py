import xarray as xr
from typing import Optional
from dataclasses import dataclass
import numpy as np

HORIZONTAL_DIMS_FV3 = ["x", "y", "tile"]
HORIZONTAL_DIMS_SCREAM = ["ncol"]
VERTICAL_DIM = "z"
PRECIP_RATE = "total_precip_to_surface"

WVP = "water_vapor_path"
COL_DRYING = "minus_column_integrated_q2"
COL_MOISTENING = "column_integrated_q2"
HISTOGRAM_BINS = {
    PRECIP_RATE: np.logspace(-1, np.log10(500), 101),
    WVP: np.linspace(-10, 90, 101),
    COL_DRYING: np.linspace(-50, 150, 101),
    COL_MOISTENING: np.linspace(-150, 50, 101),
}

# argument typehint for diags in save_prognostic_run_diags but used
# by multiple modules split out to group operations and simplify
# the diagnostic script


@dataclass
class DiagArg:
    prediction: xr.Dataset
    verification: xr.Dataset
    grid: xr.Dataset
    delp: Optional[xr.DataArray] = None
    gsrm: str = "fv3gfs"
    vertical_dim: str = VERTICAL_DIM

    def __post_init__(self):
        if self.gsrm == "fv3gfs":
            self.horizontal_dims = HORIZONTAL_DIMS_FV3
        elif self.gsrm == "scream":
            self.horizontal_dims = HORIZONTAL_DIMS_SCREAM
        else:
            raise ValueError(
                f"gsrm must be one of {['fv3gfs', 'scream']}, got {self.gsrm}"
            )
