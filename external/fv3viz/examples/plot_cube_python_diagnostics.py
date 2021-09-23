"""
plot_cube with python wrapper variables
=======================================

Example of :py:func:`plot_cube` using python wrapper output data,
with faceting over timesteps
"""

import requests
import io
import warnings
import xarray as xr
from fv3viz import plot_cube

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
    "main/fv3net/fv3viz/plot_0_plot_cube_prognostic_ml.nc"
)
GRID_URL = (
    "https://raw.githubusercontent.com/ai2cm/vcm-ml-example-data/"
    "main/fv3net/fv3viz/grid.nc"
)
VAR = "net_heating"

prognostic_ds = get_web_dataset(DATA_URL)
grid_ds = get_web_dataset(GRID_URL)
merged_ds = xr.merge([prognostic_ds, grid_ds])

_ = plot_cube(
    merged_ds, VAR, vmin=-100, vmax=100, cmap="seismic_r", col="time", col_wrap=2,
)
