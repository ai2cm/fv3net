"""
plot_cube_prognostic_ml
=======================

Example of :py:func:`plot_cube` using prognostic run python diagnostic output data,
with faceting over timesteps
"""

import os
import xarray as xr
from xarray.tutorial import open_dataset
from fv3viz import plot_cube, mappable_var
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=(
        "Tight layout not applied. The left and right margins cannot be made "
        "large enough to accommodate all axes decorations."
    ),
)

DATA_DIR = "./fv3net/fv3viz"
DATA_PATH = os.path.join(DATA_DIR, "plot_0_plot_cube_prognostic_ml.nc")
GRID_PATH = os.path.join(DATA_DIR, "grid.nc")
OPEN_DATASET_KWARGS = {
    "cache_dir": ".",
    "cache": True,
    "github_url": "https://github.com/VulcanClimateModeling/vcm-ml-example-data",
    "branch": "main",
}
VAR = "net_heating"
MAPPABLE_VAR_KWARGS = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "x_interface",
    "coord_y_outer": "y_interface",
    "coord_vars": {
        "lonb": ["y_interface", "x_interface", "tile"],
        "latb": ["y_interface", "x_interface", "tile"],
        "lon": ["y", "x", "tile"],
        "lat": ["y", "x", "tile"],
    },
}

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)

prognostic_ds = open_dataset(DATA_PATH, **OPEN_DATASET_KWARGS)
grid_ds = open_dataset(GRID_PATH, **OPEN_DATASET_KWARGS)

merged_ds = xr.merge([prognostic_ds, grid_ds])
_ = plot_cube(
    mappable_var(merged_ds, VAR, **MAPPABLE_VAR_KWARGS),
    vmin=-100,
    vmax=100,
    cmap="seismic_r",
    col="time",
    col_wrap=2,
)
