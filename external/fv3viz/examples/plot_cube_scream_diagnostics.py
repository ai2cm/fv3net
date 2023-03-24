"""
plot_cube with DOE's SCREAM diagnostics
===========================================

Example of :py:func:`plot_cube` using SCREAM diagnostic data, with faceting
over timesteps
"""

import requests
import io
import warnings
import xarray as xr
from fv3viz import plot_cube
from vcm.cubedsphere import GridMetadataScream

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
    "main/fv3net/fv3viz/plot_cube_scream_sample.nc"
)
VAR = "T_2m"


diagnostic_ds = get_web_dataset(DATA_URL)

grid_metadata = GridMetadataScream("ncol", "lon", "lat")

_ = plot_cube(
    diagnostic_ds,
    VAR,
    grid_metadata=grid_metadata,
    cmap="viridis_r",
    col="time",
    col_wrap=2,
    plotting_function="tripcolor",
    gsrm_name="scream",
)
