"""
plot_pcolormesh_cube
====================

Example of using :py:func:`plot_pcolormesh_cube` to plot a map with grid cell boundaries
"""

from matplotlib import pyplot as plt
from fv3viz import pcolormesh_cube
import intake
import cftime
from vcm.catalog import catalog
from cartopy import crs as ccrs

PATH = "gs://vcm-ml-code-testing-data/sample-prognostic-run-output/diags.zarr"
VAR = "net_moistening"
TIME = cftime.DatetimeJulian(2016, 8, 5, 4, 45, 0, 0)

prognostic_ds = intake.open_zarr(PATH).to_dask()
grid_ds = catalog["grid/c48"].to_dask()
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.Robinson()})
h = pcolormesh_cube(
    grid_ds["latb"].values,
    grid_ds["lonb"].values,
    prognostic_ds[VAR].sel(time=TIME).values,
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
