"""
Some helper function for diagnostics.

These are specifically for usage in fv3net.
There are more general purpose plotting functions in
vcm.visualize, some of which are utilized here.
"""
import matplotlib.pyplot as plt

# TODO: map plotting function is waiting on PR #67 to get merged

from vcm.visualize import plot_cube, mappable_var
from vcm.cubedsphere.constants import (
    VAR_LON_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_OUTER,
    VAR_LAT_OUTER,
    VAR_GRID_LON_CENTER,
    VAR_GRID_LAT_CENTER,
    VAR_GRID_LON_OUTER,
    VAR_GRID_LAT_OUTER,
)


TIME_VAR = "initialization_time"


def create_plot(ds, plot_config):
    for dim, dim_slice in plot_config.dim_slices.items():
        ds = ds.isel({dim: dim_slice})
    for function, kwargs in zip(plot_config.functions, plot_config.function_kwargs):
        ds = ds.pipe(function, **kwargs)

    if plot_config.plotting_function in globals():
        plot_func = globals()[plot_config.plotting_function]
        return plot_func(ds, plot_config)
    else:
        raise ValueError(
            f'Invalid plotting_function "{plot_config.plotting_function}" provided in config, \
            must correspond to function existing in fv3net.diagnostics.visualize'
        )


def plot_diag_var_map(ds, plot_config):
    """Uses vcm.visualize.plot_cube to plot a

    Args:
        ds: xr dataset
        plot_config: dict that specifies variable and dimensions to plot,
        functions to apply
        plot_func: optional plotting function from vcm.visualize
    Returns:
        matplotlib Figure object
    """
    rename_coord_vars = {
        VAR_GRID_LON_OUTER: VAR_LON_OUTER,
        VAR_GRID_LAT_OUTER: VAR_LAT_OUTER,
        VAR_GRID_LON_CENTER: VAR_LON_CENTER,
        VAR_GRID_LAT_CENTER: VAR_LAT_CENTER,
    }
    ds = ds.rename(rename_coord_vars)
    ds_mappable = mappable_var(ds, var_name=plot_config.diagnostic_variable[0])
    fig, axes, handles, cbar = plot_cube(ds_mappable, **plot_config.plot_params)
    return fig


def plot_time_series(ds, plot_config):
    fig = plt.figure()
    # allow plotting two variables on same time series
    if type(plot_config.diagnostic_variable) == str:
        plot_config.diagnostic_variable = [plot_config.diagnostic_variable]
    for var in plot_config.diagnostic_variable:
        dims_to_avg = [
            dim for dim in ds[var].dims if dim != TIME_VAR
        ]
        time = ds[TIME_VAR].values
        diag_var = ds[var].mean(dims_to_avg).values
        ax = fig.add_subplot(111)
        ax.plot(time, diag_var, label=var)
        if "xlabel" in plot_config.plot_params:
            ax.set_xlabel(plot_config.plot_params["xlabel"])
        if "ylabel" in plot_config.plot_params:
            ax.set_ylabel(plot_config.plot_params["ylabel"])
        ax.legend()
    return fig
