import matplotlib.pyplot as plt
import numpy as np


def create_plot(
        ds,
        plot_config
):
    if plot_config.plot_type=='map':
        return _plot_var_map(ds, plot_config)
    elif plot_config.plot_type=='time_series':
        return _plot_var_time_series(ds, plot_config)
    else:
        raise ValueError("Invalid plot_type in config, must be 'map', 'time_series'")

def _plot_var_map(
        ds,
        plot_config,
        plot_func=None
):
    """

    Args:
        ds: xr dataset
        plot_config: dict that specifies variable and dimensions to plot,
        functions to apply
        plot_func: optional plotting function from vcm.visualize
    Returns:
        axes
    """
    for dim, dim_slice in plot_config.dim_slices.items():
        ds = ds.isel({dim: dim_slice})
    for function, kwargs in zip(plot_config.functions, plot_config.kwargs):
        ds = ds.pipe(function, **kwargs)
    if plot_func:
        return plot_func(ds[plot_config.var])
    else:
        return ds[plot_config.var].plot()


def _plot_var_time_series(ds, plot_config):
    pass