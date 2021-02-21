"""
plot_cube_fortran_diagnostic
============================

Example of :py:func:`plot_cube` using FV3 Fortran diagnostic data, with faceting
over timesteps
"""

from fv3viz import plot_cube, mappable_var
import intake
import cftime

PATH = "gs://vcm-ml-code-testing-data/sample-prognostic-run-output/sfc_dt_atmos.zarr"
VAR = "LHTFLsfc"
TIMESTEPS = slice(
    cftime.DatetimeJulian(2016, 8, 5, 4, 45, 0, 0),
    cftime.DatetimeJulian(2016, 8, 5, 6, 0, 0, 0),
)

prognostic_ds = intake.open_zarr(PATH).to_dask()
# grid data is already present in this Fortran diagnostic file
plot_cube(
    mappable_var(prognostic_ds.sel(time=TIMESTEPS), VAR),
    vmin=-100,
    vmax=300,
    cmap="viridis_r",
    col="time",
    col_wrap=2,
)
