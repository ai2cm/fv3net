"""
plot_cube with GFDL FV3 Fortran diagnostics
===========================================

Example of :py:func:`plot_cube` using GFDL FV3 Fortran diagnostic data, with faceting
over timesteps
"""

import requests
import io
import warnings
import xarray as xr
from fv3viz import plot_cube
from vcm.cubedsphere import GridMetadata

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=(
        "Tight layout not applied. The left and right margins cannot be made "
        "large enough to accommodate all axes decorations."
    ),
)


def get_web_dataset(url):
    r = requests.get(url)
    ds = xr.open_dataset(io.BytesIO(r.content))
    return ds


DATA_URL = (
    "https://raw.githubusercontent.com/ai2cm/vcm-ml-example-data/"
    "main/fv3net/fv3viz/plot_1_plot_cube_fortran_diagnostic.nc"
)
VAR = "LHTFLsfc"


fortran_diagnostic_ds = get_web_dataset(DATA_URL)

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
