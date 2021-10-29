import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Mapping, Union, Optional
import xarray as xr

import fv3viz
from fv3net.diagnostics.offline._helpers import units_from_name


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
