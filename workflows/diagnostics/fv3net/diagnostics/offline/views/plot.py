import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from typing import Sequence, Mapping, Union, Optional
import xarray as xr

import fv3viz
from fv3net.diagnostics.offline._helpers import units_from_name


def get_plot_dataset(ds, var_filter: str, column_filters: Sequence[str]):
    # rearranges variables into dataset with domain as a dimension
    vars = [v.replace(var_filter, "").strip("_") for v in ds if var_filter in v]
    for col_tag in column_filters:
        vars = [v.replace(col_tag, "").strip("_") for v in vars]

    row_vars = set(vars)

    grouped_ds = xr.Dataset()
    for var in row_vars:
        existing_cols, grouped = [], []
        for col_tag in column_filters:
            if f"{var}_{var_filter}_{col_tag}" in ds:
                grouped.append(ds[f"{var}_{var_filter}_{col_tag}".strip("_")])
                existing_cols.append(col_tag)

        if len(grouped) > 0:
            grouped_ds[var] = xr.concat(
                grouped, dim=pd.Index(existing_cols, name="domain")
            ).squeeze()
    return grouped_ds


def plot_transect(
    data: xr.DataArray,
    xaxis: str = "lat",
    yaxis: str = "pressure",
    column_dim: str = "derivation",
    dataset_dim: str = "dataset",
):
    row_dim = dataset_dim if dataset_dim in data.dims else None
    num_datasets = len(data[dataset_dim]) if dataset_dim in data.dims else 1
    figsize = (10, 4 * num_datasets)
    facetgrid = data.plot(
        y=yaxis,
        x=xaxis,
        yincrease=False,
        col=column_dim,
        row=row_dim,
        figsize=figsize,
        robust=True,
    )
    facetgrid.set_ylabels("Pressure [Pa]")
    facetgrid.set_xlabels("Latitude [deg]")

    f = facetgrid.fig
    return f


def plot_diurnal_cycles(
    ds_diurnal: xr.Dataset, var: str, dpi: int = 100,
):
    facetgrid = ds_diurnal[var].squeeze().plot(hue="derivation", col="domain")

    facetgrid.set_titles(template="{value}", maxchar=40)
    f = facetgrid.fig
    axes = facetgrid.axes
    for ax in axes.flatten():
        ax.grid(axis="y")
        ax.set_xlabel("local_time [hrs]")
        ax.set_ylabel(units_from_name(var))
        ax.set_xlim([0, 23])
        ax.set_xticks(np.linspace(0, 24, 13))
    f.set_size_inches([12, 4])
    f.set_dpi(dpi)
    f.suptitle(var)
    f.tight_layout()
    return f


def plot_zonal_average(
    data: xr.DataArray,
    rename_axes: Mapping = None,
    title: str = None,
    plot_kwargs: Mapping = None,
):
    fig = plt.figure()
    units = units_from_name(data.name) or ""
    title = f"{title or data.name} {units}"
    plot_kwargs = plot_kwargs or {}
    rename_axes = rename_axes or {
        "latitude": "Latitude [deg]",
        "pressure": "Pressure [Pa]",
    }
    data = data.rename(rename_axes).rename(title)
    data.plot(yincrease=False, x="Latitude [deg]", robust=True, **plot_kwargs)
    return fig


def plot_profile_var(
    ds: xr.Dataset,
    var: str,
    dpi: int = 100,
    derivation_dim: str = "derivation",
    domain_dim: str = "domain",
    dataset_dim: str = "dataset",
    derivation_plot_coords: Sequence[str] = ("target", "predict"),
    xlim: Sequence[float] = None,
    xticks: Union[Sequence[float], np.ndarray] = None,
):
    if dataset_dim in ds[var].dims:
        ds[var] = ds[var].mean("dataset")
    if derivation_dim in ds[var].dims:
        facet_grid = (
            ds[var]
            .sel({derivation_dim: list(derivation_plot_coords)})
            .plot(y="z", hue=derivation_dim, col=domain_dim)
        )
    facet_grid.set_titles(template="{value}", maxchar=40)
    f = facet_grid.fig
    for ax in facet_grid.axes.flatten():
        ax.invert_yaxis()
        ax.plot([0, 0], [1, 79], "k-")
        ax.set_xlabel(f"{var} {units_from_name(var)}")
        if xlim:
            ax.set_xlim(xlim)
    f.set_size_inches([17, 3.5])
    f.set_dpi(dpi)
    f.suptitle(var.replace("_", " "))
    return f


def plot_column_integrated_var(
    ds: xr.Dataset,
    var: str,
    derivation_plot_coords: Optional[Sequence[str]] = None,
    derivation_dim: Optional[str] = "derivation",
    dataset_dim: str = "dataset",
    dpi: int = 100,
    vmax: Union[int, float] = None,
):
    ds_columns = (
        ds.sel(derivation=derivation_plot_coords)
        if derivation_plot_coords is not None
        else ds
    )
    f, _, _, _, facet_grid = fv3viz.plot_cube(
        ds_columns,
        var,
        col=derivation_dim,
        row=dataset_dim if dataset_dim in ds.dims else None,
        vmax=vmax,
    )
    if facet_grid:
        facet_grid.set_titles(template="{value} ", maxchar=40)
    num_datasets = len(ds[dataset_dim]) if dataset_dim in ds.dims else 1
    f.set_size_inches([14, 3.5 * num_datasets])
    f.set_dpi(dpi)
    f.suptitle(f'{var.replace("_", " ")} {units_from_name(var)}')
    return f


def plot_generic_data_array(
    da: xr.DataArray,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    xlim: Sequence[float] = None,
    ylim: Sequence[float] = None,
):
    fig = plt.figure()
    da.plot()
    if xlabel:
        plt.xlabel(xlabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    units = units_from_name(da.name) or ""
    ylabel = ylabel or units
    title = title or " ".join([da.name.replace("_", " ").replace("-", ",")])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_histogram(ds, varname: str, xscale="linear", yscale="linear"):
    """Plot 1D histogram of varname."""

    fig, ax = plt.subplots()
    bin_name = varname.replace("histogram", "bins")
    v = ds[varname]
    units = units_from_name(varname.replace("_histogram", ""))
    ax.step(v[bin_name], v, where="post", linewidth=1)
    ax.set_xlabel(f"{getattr(v, 'long_name', v.name)} {units}")
    ax.set_ylabel(f"Frequency {units}^-1")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_ylim([0, None])
    ax.legend()
    fig.tight_layout()
    return fig


def plot_histogram2d(ds, xname: str, yname: str):
    """Plot 2D histogram of xname versus yname."""

    count_name = f"{xname.lower()}_versus_{yname.lower()}_hist_2d"
    conditional_average_name = f"conditional_average_of_{yname}_on_{xname}"
    x_bin_name = f"{xname}_bins"
    y_bin_name = f"{yname}_bins"
    x_bin_widths_name = f"{xname.lower()}_bin_width_hist_2d"
    y_bin_widths_name = f"{yname.lower()}_bin_width_hist_2d"

    count = ds[count_name]
    conditional_average = ds[conditional_average_name]
    x_bin_widths = ds[x_bin_widths_name]
    y_bin_widths = ds[y_bin_widths_name]
    x = x_bin_widths[x_bin_name]
    y = y_bin_widths[y_bin_name]
    xedges = np.append(x.values, x.values[-1] + x_bin_widths.values[-1])
    yedges = np.append(y.values, y.values[-1] + y_bin_widths.values[-1])
    xcenters = x.values + 0.5 * x_bin_widths.values
    fig, ax = plt.subplots()
    xx, yy = np.meshgrid(xedges, yedges)
    ax.pcolormesh(
        xx, yy, count.T, norm=mpl.colors.LogNorm(), cmap="Blues", rasterized=True
    )
    ax.plot(xcenters, conditional_average, color="r", linewidth=2)

    x_units = units_from_name(x_bin_widths.name.replace("_bin_width_hist_2d", ""))
    y_units = units_from_name(y_bin_widths.name.replace("_bin_width_hist_2d", ""))
    ax.set_xlabel(f"{xname} {x_units}")
    ax.set_ylabel(f"{yname} {y_units}")
    ax.set_xlim([xedges[0], xedges[-1]])
    ax.set_ylim([yedges[0], yedges[-1]])
    plt.tight_layout()
    return fig
