"""
plot_cube_prognostic_report
===========================

Example of :py:func:`plot_cube` using a time-mean prognostic run report diag
"""

import xarray as xr
from fv3viz import plot_cube, mappable_var
import fsspec
from matplotlib import pyplot as plt

PATH = (
    "gs://vcm-ml-archive/prognostic_run_diags/"
    "vcm-ml-experiments-2021-01-22-nudge-to-fine-3h-prognostic_run/diags.nc"
)
VAR = "h500_time_mean_value"
MAPPABLE_VAR_KWARGS = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "xb",
    "coord_y_outer": "yb",
    "coord_vars": {
        "lonb": ["yb", "xb", "tile"],
        "latb": ["yb", "xb", "tile"],
        "lon": ["y", "x", "tile"],
        "lat": ["y", "x", "tile"],
    },
}

with fsspec.open(PATH, "rb") as f:
    prognostic_ds = xr.open_dataset(f).load()
args = plot_cube(
    mappable_var(prognostic_ds, VAR, **MAPPABLE_VAR_KWARGS), vmin=5000, vmax=5900
)
args[0].set_size_inches([10, 4])
plt.tight_layout()
prognostic_ds.close()
