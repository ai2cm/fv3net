"""
plot_cube_prognostic_ml
=======================

Example of :py:func:`plot_cube` using prognostic run python diagnostic output data,
with faceting over timesteps
"""

import xarray as xr
from fv3viz import plot_cube, mappable_var
from vcm.catalog import catalog
import intake
import cftime

PATH = "gs://vcm-ml-code-testing-data/sample-prognostic-run-output/diags.zarr"
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
TIMESTEPS = slice(
    cftime.DatetimeJulian(2016, 8, 5, 4, 45, 0, 0),
    cftime.DatetimeJulian(2016, 8, 5, 6, 0, 0, 0),
)

prognostic_ds = intake.open_zarr(PATH).to_dask()
grid_ds = catalog["grid/c48"].to_dask()
# subset data in time and merge with grid
subset_ds = xr.merge([prognostic_ds.sel(time=TIMESTEPS), grid_ds])
plot_cube(
    mappable_var(subset_ds, VAR, **MAPPABLE_VAR_KWARGS),
    vmin=-100,
    vmax=100,
    cmap="seismic_r",
    col="time",
    col_wrap=2,
)
