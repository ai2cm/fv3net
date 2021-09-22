"""
plot_cube with GFDL FV3 Fortran diagnostics
===========================================

Example of :py:func:`plot_cube` using GFDL FV3 Fortran diagnostic data, with faceting
over timesteps
"""

import os
from xarray.tutorial import open_dataset
from fv3viz import plot_cube
from vcm.cubedsphere import GridMetadata
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
DATA_PATH = os.path.join(DATA_DIR, "plot_1_plot_cube_fortran_diagnostic.nc")
OPEN_DATASET_KWARGS = {
    "cache_dir": ".",
    "cache": True,
    "github_url": "https://github.com/VulcanClimateModeling/vcm-ml-example-data",
    "branch": "main",
}
VAR = "LHTFLsfc"

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)

fortran_diagnostic_ds = open_dataset(DATA_PATH, **OPEN_DATASET_KWARGS)
gfdl_grid_metadata = GridMetadata("grid_xt", "grid_yt", "grid_x", "grid_y")

# grid variables are already present in this Fortran diagnostic file
_ = plot_cube(
    fortran_diagnostic_ds,
    VAR,
    grid_metadata=gfdl_grid_metadata,
    vmin=-100,
    vmax=300,
    cmap="viridis_r",
    col="time",
    col_wrap=2,
)
