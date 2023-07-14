"""Visualize some fields and compare to a trivial nearest-neighbor baseline."""

import xarray as xr
import numpy as np
import fv3viz
from vcm.catalog import catalog
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs

GRID = catalog["grid/c384"].read()

DPI = 800


def repeat_arr(arr, upscale_factor):
    # This repeats values along the last dimension
    return np.repeat(arr, upscale_factor, axis=-1)


def upscale_nearest_neighbor(ds: xr.Dataset, upscale_factor: int, dim: str):
    ds_upscaled = xr.apply_ufunc(
        repeat_arr,
        ds,
        input_core_dims=[[dim]],  # apply function along this dimension
        output_core_dims=[[dim]],  # the output will also have this dimension
        kwargs={
            "upscale_factor": upscale_factor
        },  # pass in other arguments to the function
        dask="parallelized",  # needed for dask arrays
        output_sizes={
            dim: ds.sizes[dim] * upscale_factor
        },  # the size of the output dimension
        exclude_dims=set((dim,)),  # allow changing the dimension size
    )
    return ds_upscaled


def plot_pattern_bias(output: xr.Dataset, target: xr.Dataset, varname: str):
    """Plot the pattern bias between two datasets

    Args:
        output (xr.Dataset): output dataset
        target (xr.Dataset): target dataset
    """
    # compute the pattern bias
    out_mean: xr.Dataset = output.mean("time")
    target_mean: xr.Dataset = target.mean("time")
    bias: xr.Dataset = out_mean - target_mean
    vmin = min(
        # must convert min() output dask array to float explicitly,
        # or we get an exception when it's used in plot_cube
        out_mean[varname].min().values.item(),
        target_mean[varname].min().values.item(),
    )
    vmax = max(
        out_mean[varname].max().values.item(), target_mean[varname].max().values.item(),
    )
    fig, ax = plt.subplots(
        3, 1, figsize=(6, 9), subplot_kw={"projection": ccrs.Robinson()}
    )
    # plot the pattern bias
    fv3viz.plot_cube(
        out_mean.merge(GRID, compat="override"),
        var_name=varname,
        ax=ax[0],
        vmin=vmin,
        vmax=vmax,
    )
    ax[0].set_title("Time-mean Output")
    fv3viz.plot_cube(
        target_mean.merge(GRID, compat="override"),
        var_name=varname,
        ax=ax[1],
        vmin=vmin,
        vmax=vmax,
    )
    ax[1].set_title("Time-mean Target")
    fv3viz.plot_cube(
        # must merge grid data into the dataset
        # after performing bias calculation on dataset
        bias.merge(GRID, compat="override"),
        var_name=varname,
        cmap="RdBu_r",
        ax=ax[2],
    )
    # TODO: add area-weighting to bias calculation
    pattern_rmse = np.sqrt((bias[varname] ** 2).mean().values.item())
    pattern_mean = bias[varname].mean().values.item()
    ax[2].set_title(
        "Pattern Bias, mean: {:.2f}, RMSE: {:.2f}".format(pattern_mean, pattern_rmse)
    )
    plt.tight_layout()
    fig.savefig(f"plots/{varname}-pattern_bias.png", dpi=DPI)


def plot_first_snapshot(output: xr.Dataset, target: xr.Dataset, varname: str):
    # must specify projection for subplots if calling plot_cube with an ax argument
    fig, ax = plt.subplots(
        3, 1, figsize=(6, 9), subplot_kw={"projection": ccrs.Robinson()}
    )
    output_snapshot = output.isel(time=0, drop=True)
    target_snapshot = target.isel(time=0, drop=True)
    vmin = min(
        output_snapshot[varname].min().values.item(),
        target_snapshot[varname].min().values.item(),
    )
    vmax = max(
        output_snapshot[varname].max().values.item(),
        target_snapshot[varname].max().values.item(),
    )
    fv3viz.plot_cube(
        output_snapshot.merge(GRID, compat="override"),
        var_name=varname,
        ax=ax[0],
        vmin=vmin,
        vmax=vmax,
    )
    fv3viz.plot_cube(
        target_snapshot.merge(GRID, compat="override"),
        var_name=varname,
        ax=ax[1],
        vmin=vmin,
        vmax=vmax,
    )
    fv3viz.plot_cube(
        (output_snapshot - target_snapshot).merge(GRID, compat="override"),
        var_name=varname,
        ax=ax[2],
    )
    ax[0].set_title("Snapshot Output")
    ax[1].set_title("Snapshot Target")
    ax[2].set_title("Snapshot Error")
    plt.tight_layout()
    fig.savefig(f"plots/{varname}-first_snapshot.png", dpi=DPI)


def plot_differences_histogram(
    output: xr.Dataset, target: xr.Dataset, upscaled_coarse: xr.Dataset, varname: str
):
    """
    Plot the histogram of differences between the output and coarse gridcell-mean.

    Args:
        output: upscaled dataset
        target: target fine-resolution dataset
        upscaled_coarse: coarse-resolution dataset upscaled to target resolution
            with nearest-neighbor sampling
        varname: name of variable to plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # compute the difference between the output and the coarse gridcell-mean
    # and convert to mm/day
    diff_output = (output - upscaled_coarse)[varname].values.flatten()
    diff_target = (target - upscaled_coarse)[varname].values.flatten()
    vmin = min(diff_output.min(), diff_target.min())
    vmax = max(diff_output.max(), diff_target.max())
    bins = np.linspace(vmin, vmax, 100 + 1)  # add one because these are bin edges
    # plot the histogram
    ax.hist(
        diff_output, bins=bins, alpha=0.5, label="Output", histtype="step", density=True
    )
    ax.hist(
        diff_target, bins=bins, alpha=0.5, label="Target", histtype="step", density=True
    )
    ax.set_xlim(vmin, vmax)
    ax.legend()
    ax.set_xlabel("Difference from gridcell mean")
    ax.set_ylabel("Density")
    ax.set_title(f"Histogram of {varname} differences from coarse cell mean")
    plt.tight_layout()
    plt.yscale("log")
    fig.savefig(f"plots/{varname}-diff_histogram.png", dpi=DPI)


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    c384 = xr.open_zarr(
        "gs://vcm-ml-raw-flexible-retention/2021-07-19-PIRE/C3072-to-C384-res-diagnostics/pire_atmos_phys_3h_coarse.zarr"  # noqa: E501
    ).rename({"grid_xt_coarse": "x", "grid_yt_coarse": "y"})
    c48 = xr.open_zarr(
        "gs://vcm-ml-intermediate/2021-10-12-PIRE-c48-post-spinup-verification/pire_atmos_phys_3h_coarse.zarr"  # noqa: E501
    ).rename({"grid_xt": "x", "grid_yt": "y"})
    # use the function
    c384_nearest = upscale_nearest_neighbor(
        upscale_nearest_neighbor(c48, 8, "x"), 8, "y"
    )
    plot_names = [
        "tsfc_coarse",
        "PRATEsfc_coarse",
        "LHTFLsfc_coarse",
        "SHTFLsfc_coarse",
        "ULWRFtoa_coarse",
        "DSWRFsfc_coarse",
    ]
    drop_vars = set(c384_nearest.data_vars) - set(plot_names)
    c384_nearest = c384_nearest.drop_vars(drop_vars).load()
    for varname in plot_names:
        print(f"plotting for {varname}")
        print("plotting first snapshot")
        plot_first_snapshot(c384_nearest, c384, varname=varname)
        print("plotting pattern bias")
        plot_pattern_bias(c384_nearest, c384, varname=varname)
        print("plotting histogram of differences from cell-mean")
        plot_differences_histogram(
            c384_nearest, c384, upscaled_coarse=c384_nearest, varname=varname
        )
