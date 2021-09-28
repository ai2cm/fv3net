"""
plot_pcolormesh_cube
====================

Example of using :py:func:`plot_pcolormesh_cube` to plot a map with grid cell boundaries
"""


import requests
import io
import warnings
import xarray as xr
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from fv3viz import pcolormesh_cube

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
    "main/fv3net/fv3viz/plot_4_plot_pcolormesh_cube.nc"
)
GRID_URL = (
    "https://raw.githubusercontent.com/ai2cm/vcm-ml-example-data"
    "/main/fv3net/fv3viz/grid.nc"
)
VAR = "net_moistening"

prognostic_ds = get_web_dataset(DATA_URL)
grid_ds = get_web_dataset(GRID_URL)

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.Robinson()})
h = pcolormesh_cube(
    grid_ds["latb"].values,
    grid_ds["lonb"].values,
    prognostic_ds[VAR].values,
    vmin=-1.0e-4,
    vmax=1.0e-4,
    cmap="seismic_r",
    ax=ax,
    edgecolor="gray",
    linewidth=0.01,
)
ax.set_title(VAR)
ax.coastlines()
plt.colorbar(h, ax=ax, label="kg/s/m^2")
fig.set_size_inches([10, 4])
fig.set_dpi(100)
