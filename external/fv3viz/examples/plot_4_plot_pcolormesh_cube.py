"""
plot_pcolormesh_cube
====================

Example of using :py:func:`plot_pcolormesh_cube` to plot a map with grid cell boundaries
"""

import os
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from xarray.tutorial import open_dataset
from fv3viz import pcolormesh_cube

DATA_DIR = "./fv3net/fv3viz"
DATA_PATH = os.path.join(DATA_DIR, "plot_4_plot_pcolormesh_cube.nc")
GRID_PATH = os.path.join(DATA_DIR, "grid.nc")
OPEN_DATASET_KWARGS = {
    "cache_dir": ".",
    "cache": True,
    "github_url": "https://github.com/VulcanClimateModeling/vcm-ml-example-data",
}
VAR = "net_moistening"

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)

prognostic_ds = open_dataset(DATA_PATH, **OPEN_DATASET_KWARGS)
grid_ds = open_dataset(GRID_PATH, **OPEN_DATASET_KWARGS)

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
fig.savefig("plot_4.png")
