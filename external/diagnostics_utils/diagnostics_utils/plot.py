import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Mapping, Sequence
import xarray as xr

import fv3viz as visualize


# grid info for the plot_cube function
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


def plot_diurnal_components(
        ds: xr.Dataset,
        diurnal_vars: Sequence[str],
        title: str,
        x_dim: str,
        derivation_dim: str,
        units: str = None,
        selection: Mapping[str, str] = None,
        output_dir: str = None,
):
    fig = plt.figure()
    time_coords = ds[diurnal_vars[0]][x_dim].values
    selection = selection or {}
    for diurnal_var in diurnal_vars:
        for derivation_coord in ds[derivation_dim].values:
            if derivation_dim in ds[diurnal_var].dims:
                selection[derivation_dim] = derivation_coord
            plt.plot(
                time_coords,
                ds[diurnal_var].sel(selection),
                label=f"{diurnal_var}, {derivation_coord}".replace("_", " ")
                )
    plt.xlabel("local time [hr]")
    plt.ylabel(units or "")
    plt.legend()
    fig.savefig(
        os.path.join(output_dir or "",
        f'{title.replace(" ", "_")}.png'))
        

def plot_profile_vars(
        ds: xr.Dataset,
        output_dir: str,
        profile_vars: Sequence[str],
        dpi: int = 100,
        derivation_dim: str = "derivation",
        domain_dim: str="domain",
        units_q1: str = "K/s",
        units_q2: str = "kg/kg/s",
        ):
    for var in profile_vars:
        if "derivation" in ds[var].dims:
            facet_grid = ds[var].plot(y='z', hue=derivation_dim, col=domain_dim)
        facet_grid.set_titles(template="{value}", maxchar=40)
        f = facet_grid.fig
        for ax in facet_grid.axes.flatten():
            ax.invert_yaxis()
            ax.plot([0, 0], [1, 79], 'k-')
            if '1' in var:
                ax.set_xlim([-0.0001, 0.0001])
                ax.set_xticks(np.arange(-1e-4, 1.1e-4, 5e-5))
            else:
                ax.set_xlim([-1e-7, 1e-7])
                ax.set_xticks(np.arange(-1e-7, 1.1e-7, 5e-8))
            ax.set_xlabel(f"{var} {_units_from_var(var)}")
        f.set_size_inches([17, 3.5])
        f.set_dpi(dpi)
        f.suptitle(var.replace("_", " "))
        f.savefig(os.path.join(output_dir, f'{var}_profile_plot.png'))


def plot_column_integrated_vars(
        ds: xr.Dataset,
        output_dir: str,
        column_integrated_vars: Sequence[str],
        derivation_plot_coords: Sequence[str],
        derivation_dim: str = "derivation",
        data_source_dim: str = None,
        dpi: int = 100
        ):
    for var in column_integrated_vars:
        
        f, _, _, _, facet_grid = visualize.plot_cube(
            visualize.mappable_var(
                ds.sel(derivation=derivation_plot_coords),
                var,
                **MAPPABLE_VAR_KWARGS
            ),
            col=derivation_dim,
            row=data_source_dim,
            vmax=(1000 if '1' in var else 10)
        )
        facet_grid.set_titles(template="{value}", maxchar=40)
        f.set_size_inches([14,3.5])
        f.set_dpi(dpi)
        f.suptitle(var.replace("_", " "))
        f.savefig(os.path.join(output_dir, f'{var}_map.png'))


def plot_diurnal_cycles(
        ds_diurnal: xr.Dataset,
        output_dir: str,
        tag: str,
        vars: Sequence[str],
        derivation_plot_coords: Sequence[str],
        dpi: int = 100,
):
    ds_diurnal = ds_diurnal.sel(derivation=derivation_plot_coords)
    facetgrid = ds_diurnal[vars].squeeze().to_array() \
            .plot(hue='derivation', row='variable', col='surface_type')
    facetgrid.set_titles(template="{value}", maxchar=40)
    f = facetgrid.fig
    axes = facetgrid.axes
    for ax in axes.flatten():
        ax.grid(axis='y')
        ax.set_xlabel('local_time [hrs]')
        ax.set_ylabel(_units_from_var(vars[0]))
        ax.set_xlim([0, 23])
        ax.set_xticks(np.linspace(0, 24, 13))
    f.set_size_inches([12, 4 * len(vars)])
    f.set_dpi(dpi)
    f.tight_layout()
    f.savefig(os.path.join(output_dir, f"{tag}_diurnal_cycle.png"))


def _plot_generic_data_array(
        da: xr.DataArray,
        output_dir: str,
        tag: str = None,
        xlabel: str=None,
        ylabel: str=None,):
    plt.figure()
    da.plot()
    if xlabel:
        plt.xlabel(xlabel)
    ylabel = ylabel or da.name.replace("_", " ").replace("-", ",")
    plt.ylabel(ylabel)
    if tag:
        tag += "_"
    plt.savefig(os.path.join(output_dir, f"{tag or ''}{da.name}.png"))


def _units_from_var(var):
    if "Q1" in var:
        if "column_integrated" in var:
            return "W/m^2"
        else:
            return "K/s"
    elif "Q2" in var:
        if "column_integrated" in var:
            return "mm/day"
        else:
            return "kg/kg/s"
    else:
        raise ValueError("Can only parse units from variables with Q1, Q2 in name.")
    