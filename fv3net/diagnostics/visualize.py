"""
Some helper function for diagnostics.

These are specifically for usage in fv3net.
There are more general purpose plotting functions in
vcm.visualize, some of which are utilized here.
"""
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np

# TODO: map plotting function is waiting on PR #67 to get merged
# from vcm.visualize import plot_cube


# Unclear if these coordinates will get new names in future so left as global vars
TIME_VAR = 'initialization_time'
VERTICAL_GRID_VAR = 'pfull'
LON_GRID_CENTER = 'grid_lont'
LAT_GRID_CENTER = 'grid_latt'
LON_GRID_EDGE = 'grid_lon'
LAT_GRID_EDGE = 'grid_lat'


def create_plot(
        ds,
        plot_config
):
    for dim, dim_slice in plot_config.dim_slices.items():
        ds = ds.isel({dim: dim_slice})
    for function, kwargs in zip(plot_config.functions, plot_config.function_kwargs):
        ds = ds.pipe(function, **kwargs)

    if plot_config.plotting_function in globals():
        plot_func = globals()[plot_config.plotting_function]
        return plot_func(ds, plot_config, **plot_config.plot_kwargs)
    else:
        raise ValueError(
            f'Invalid plotting_function "{plot_config.plotting_function}" provided in config, \
            must correspond to function existing in fv3net.diagnostics.visualize')


def plot_diag_var_map(
        ds,
        plot_config,
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
    grid = ds[[LAT_GRID_EDGE, LON_GRID_EDGE, LAT_GRID_CENTER, LON_GRID_CENTER]]
    fig, ax = plot_cube(
        ds[plot_config.diagnostic_variable],
        grid)
    return fig


def plot_snapshots(
        ds,
        plot_config
):
    pass


def plot_time_series(
        ds,
        plot_config,
        xlabel=None,
        ylabel=None
):
    dims_to_avg = [
        dim for dim in ds[plot_config.diagnostic_variable].dims
        if dim != TIME_VAR]
    time = ds[TIME_VAR].values
    diag_var = ds[plot_config.diagnostic_variable].mean(dims_to_avg).values
    fig = plt.plot(
        time,
        diag_var)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    return fig


def plot_histogram(ds, plot_config):
    pass


"""
These seem to be older unused functions that didn't get cleaned up in a previous refactor?
"""


def make_image(
        sliced_data,
        cmap_range=None,
        coords=None,
        quad=False,
        invert_y=False,
        **kwargs
):
    coords = coords or (
        ["grid_xt", "grid_yt"]
        if "grid_xt" in sliced_data.coords
        else ["lon", "lat"]
    )
    hv_img = (
        hv.QuadMesh(sliced_data, coords)
        if quad
        else hv.Image(sliced_data, coords)
    )
    if cmap_range is not None:
        var_name = sliced_data.name
        hv_img = hv_img.redim(
            **{var_name: hv.Dimension(var_name, range=cmap_range)}
        )
    hv_img = hv_img.options(**kwargs)
    return hv_img


def make_animation(
        sliced_data,
        cmap_range=None,
        coords=None,
        quad=False,
        invert_y=False,
        **kwargs
):
    coords = coords or (
        ["grid_xt", "grid_yt"]
        if "grid_xt" in sliced_data.coords
        else ["lon", "lat"]
    )
    hv_ds = hv.Dataset(sliced_data)
    hv_img = hv_ds.to(hv.QuadMesh if quad else hv.Image, coords).options(
        **kwargs
    )
    if cmap_range is not None:
        var_name = sliced_data.name
        hv_img = hv_img.redim(
            **{var_name: hv.Dimension(var_name, range=cmap_range)}
        )
    return hv_img
