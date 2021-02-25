"""
plot_cube_prognostic_diags
==========================

Example of :py:func:`plot_cube` using a time-mean prognostic run report diag
"""

import os
from matplotlib import pyplot as plt
from xarray.tutorial import open_dataset
from fv3viz import plot_cube, mappable_var

DATA_DIR = "./fv3net/fv3viz"
DATA_PATH = os.path.join(DATA_DIR, "plot_2_plot_cube_prognostic_diags.nc")
OPEN_DATASET_KWARGS = {
    "cache_dir": ".",
    "cache": True,
    "github_url": "https://github.com/VulcanClimateModeling/vcm-ml-example-data",
}
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

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)

prognostic_diags_ds = open_dataset(DATA_PATH, **OPEN_DATASET_KWARGS)

# grid data is already present in this prognostic run diagnostics file
args = plot_cube(
    mappable_var(prognostic_diags_ds, VAR, **MAPPABLE_VAR_KWARGS), vmin=5000, vmax=5900
)
args[0].set_size_inches([10, 4])
plt.tight_layout()
