"""
Some helper functions for creating diagnostic plots.

These are specifically for usage in fv3net.

Uses the general purpose plotting functions in
vcm.visualize such as plot_cube.


"""
import matplotlib.pyplot as plt
import warnings

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

# TODO: this may have to change if we change to using >1 step runs
TIME_VAR = "initialization_time"


def create_plot(ds, plot_config):
    """ Subselects data, pipes through functions to produce final diagnostic quantity,
    and the passes to appropriate plotting function.

    Args:
        ds: xarray dataset
        plot_config: PlotConfig object

    Returns:
        matplotlib figure
    """
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
    """ Uses vcm.visualize.plot_cube to plot

    Args:
        ds: xr dataset
        plot_config: PlotConfig object
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
    if len(plot_config.diagnostic_variable) > 1:
        warnings.warn(
            "More than one diagnostic variable provided in config entry "
            "for a map plot figure, check your config. Only plotting the "
            "first diagnostic variable in the list."
        )
    ds_mappable = mappable_var(ds, var_name=plot_config.diagnostic_variable[0])
    fig, axes, handles, cbar = plot_cube(ds_mappable, **plot_config.plot_params)
    return fig


def plot_time_series(ds, plot_config):
    """ Plot one or more variables as a time series.

    Args:
        ds: xarray dataset
        plot_config: PlotConfig object

    Returns:
        matplotlib figure
    """
    fig = plt.figure()
    for var in plot_config.diagnostic_variable:
        time = ds[TIME_VAR].values
        ax = fig.add_subplot(111)
        ax.plot(time, ds[var].values, label=var)
        if "xlabel" in plot_config.plot_params:
            ax.set_xlabel(plot_config.plot_params["xlabel"])
        if "ylabel" in plot_config.plot_params:
            ax.set_ylabel(plot_config.plot_params["ylabel"])
        ax.legend()
    return fig
