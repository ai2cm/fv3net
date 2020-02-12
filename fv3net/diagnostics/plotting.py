"""
Some helper functions for creating diagnostic plots.

These are specifically for usage in fv3net.

Uses the general purpose plotting functions in
vcm.visualize such as plot_cube.


"""
import matplotlib.pyplot as plt
import os
from scipy.stats import binned_statistic
import xarray as xr

from vcm.visualize import plot_cube, mappable_var
from vcm.cubedsphere.constants import (
    INIT_TIME_DIM,
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_Z_CENTER,
    PRESSURE_GRID,
)

STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]


def plot_diurnal_cycle(
    merged_ds,
    var,
    output_dir,
    num_time_bins=24,
    title=None,
    plot_filename="diurnal_cycle.png",
    save_fig=True,
):
    """

    Args:
        merged_ds: xr dataset, can either provide a merged dataset with a "dataset" dim
        that will be used to plot separate lines for each variable, or a single dataset
        with no "dataset" dim
        var: name of variable to plot
        output_dir: write location for figure
        num_time_bins: number of bins per day
        title: optional plot title
        dataset_labels: optional list of str labels for datasets, in order corresponding
            to datasets_to_plot
        plot_filename: filename to save to

    Returns:
        None
    """
    plt.clf()
    for label in merged_ds["dataset"].values:
        ds = merged_ds.sel(dataset=label).stack(sample=STACK_DIMS).dropna("sample")
        local_time = ds["local_time"].values.flatten()
        data_var = ds[var].values.flatten()
        bin_means, bin_edges, _ = binned_statistic(
            local_time, data_var, bins=num_time_bins
        )
        bin_centers = [
            0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(num_time_bins)
        ]
        plt.plot(bin_centers, bin_means, label=label)
    plt.xlabel("local_time [hr]")
    plt.ylabel(var)
    plt.legend(loc="lower left")
    if title:
        plt.title(title)
    if save_fig:
        plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()



# functions below here are from the previous design and probably outdated
# leaving for now as they might be adapted to work with new design

def plot_diag_var_single_map(
        da,
        grid,
        var_name,
        plot_cube_kwargs=None):
    """ Uses vcm.visualize.plot_cube to plot

    Args:
        da: xr data array of variable to plot
        grid: xr dataset with grid coordinate variables on the same dims as da
        var_name: name of variable
    Returns:
        matplotlib Figure object
    """
    da = da.rename(var_name)
    ds_mappable = mappable_var(xr.merge([da, grid]), var_name)
    fig, axes, handles, cbar = plot_cube(ds_mappable, **(plot_cube_kwargs or {}))
    return fig


def plot_time_series(
        ds,
        vars_to_plot,
        output_dir,
        time_var=INIT_TIME_DIM,
        plot_kwargs=None,
        plot_filename="time_series.png",
):
    """ Plot one or more variables as a time series.

    Args:
        ds: xarray dataset
        plot_config: PlotConfig object

    Returns:
        matplotlib figure
    """
    plt.clf()
    plot_kwargs = plot_kwargs or {}
    for var in vars_to_plot:
        time = ds[time_var].values
        plt.plot(time, ds[var].values, label=var)
        if "xlabel" in plot_kwargs:
            plt.xlabel(plot_kwargs["xlabel"])
        if "ylabel" in plot_kwargs:
            plt.ylabel(plot_kwargs["ylabel"])
        plt.legend()
    if "title" in plot_kwargs:
        plt.title(plot_kwargs["title"])
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.show()
