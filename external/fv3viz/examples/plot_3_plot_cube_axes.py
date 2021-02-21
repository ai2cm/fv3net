"""
plot_cube_axes
==============

Example of using :py:func:`plot_cube_axes` to plot two different variables on the
same figure
"""

import xarray as xr
from matplotlib import pyplot as plt
from fv3viz import plot_cube_axes, mappable_var
import intake
import cftime
from vcm.catalog import catalog
from cartopy import crs as ccrs

PATH = "gs://vcm-ml-code-testing-data/sample-prognostic-run-output/diags.zarr"
VAR1 = "net_moistening"
VAR2 = "net_heating"
TIME = cftime.DatetimeJulian(2016, 8, 5, 4, 45, 0, 0)
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

prognostic_ds = intake.open_zarr(PATH).to_dask()
grid_ds = catalog["grid/c48"].to_dask()
ds1 = mappable_var(
    xr.merge([prognostic_ds[VAR1].sel(time=TIME), grid_ds]), VAR1, **MAPPABLE_VAR_KWARGS
).load()
fig, axes = plt.subplots(2, 1, subplot_kw={"projection": ccrs.Robinson()})
h1 = plot_cube_axes(
    ds1[VAR1].values,
    ds1["lat"].values,
    ds1["lon"].values,
    ds1["latb"].values,
    ds1["lonb"].values,
    "pcolormesh",
    vmin=-1.0e-4,
    vmax=1.0e-4,
    cmap="seismic_r",
    ax=axes[0],
)
axes[0].set_title(VAR1)
axes[0].coastlines()
plt.colorbar(h1, ax=axes[0], label="kg/s/m^2")
ds2 = mappable_var(
    xr.merge([prognostic_ds[VAR2].sel(time=TIME), grid_ds]), VAR2, **MAPPABLE_VAR_KWARGS
).load()
h2 = plot_cube_axes(
    ds2[VAR2].values,
    ds2["lat"].values,
    ds2["lon"].values,
    ds2["latb"].values,
    ds2["lonb"].values,
    "pcolormesh",
    vmin=-1.0e2,
    vmax=1.0e2,
    cmap="seismic_r",
    ax=axes[1],
)
axes[1].set_title(VAR2)
axes[1].coastlines()
plt.colorbar(h2, ax=axes[1], label="W/m^2")
fig.set_size_inches([8, 8])
