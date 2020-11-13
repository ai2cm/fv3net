import argparse
import logging
import os
from multiprocessing import get_context
from typing import Sequence, Tuple

import dask


import intake
import fsspec
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import fv3viz as viz
import vcm
import vcm.catalog
import config
import load_diagnostic_data as load_diags

dask.config.set(sheduler="single-threaded")


MovieArg = Tuple[xr.Dataset, str]
FIG_SUFFIX = "_{t:05}.png"

COORD_NAMES = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "xb",
    "coord_y_outer": "yb",
}

_COORD_VARS = {
    "lonb": ["yb", "xb", "tile"],
    "latb": ["yb", "xb", "tile"],
    "lon": ["y", "x", "tile"],
    "lat": ["y", "x", "tile"],
}

GRID_VARS = ["area", "lonb", "latb", "lon", "lat"]
INTERFACE_DIMS = ["xb", "yb"]

HEATING_MOISTENING_PLOT_KWARGS = {
    "column_integrated_pQ1": {"vmin": -600, "vmax": 600, "cmap": "RdBu_r"},
    "column_integrated_dQ1": {"vmin": -600, "vmax": 600, "cmap": "RdBu_r"},
    "column_integrated_Q1": {"vmin": -600, "vmax": 600, "cmap": "RdBu_r"},
    "column_integrated_pQ2": {"vmin": -20, "vmax": 20, "cmap": "RdBu_r"},
    "column_integrated_dQ2": {"vmin": -20, "vmax": 20, "cmap": "RdBu_r"},
    "column_integrated_Q2": {"vmin": -20, "vmax": 20, "cmap": "RdBu_r"},
}

WIND_TENDENCY_PLOT_KWARGS = {
    "vertical_mean_dQu": {"vmin": -5, "vmax": 5, "cmap": "RdBu_r"},
    "vertical_mean_dQv": {"vmin": -5, "vmax": 5, "cmap": "RdBu_r"},
}


def _plot_maps(ds, axes, plot_kwargs):
    for i, (variable, variable_plot_kwargs) in enumerate(plot_kwargs.items()):
        ax = axes.flatten()[i]
        mv = viz.mappable_var(ds, variable, coord_vars=_COORD_VARS, **COORD_NAMES)
        viz.plot_cube(mv, ax=ax, **variable_plot_kwargs)
        ax.set_title(variable.replace("_", " "))


def _save_heating_moistening_fig(arg: MovieArg):
    ds, fig_filename = arg
    print(f"Saving to {fig_filename}")
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 5.3), subplot_kw={"projection": ccrs.Robinson()}
    )
    _plot_maps(ds, axes, HEATING_MOISTENING_PLOT_KWARGS)
    fig.suptitle(ds.time.values.item())
    plt.subplots_adjust(left=0.01, right=0.91, bottom=0.05, wspace=0.32)
    with fsspec.open(fig_filename, "wb") as fig_file:
        fig.savefig(fig_file, dpi=100)
    plt.close(fig)


def _save_wind_tendency_fig(arg: MovieArg):
    ds, fig_filename = arg
    print(f"Saving to {fig_filename}")
    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 3), subplot_kw={"projection": ccrs.Robinson()}
    )
    _plot_maps(ds, axes, WIND_TENDENCY_PLOT_KWARGS)
    fig.suptitle(ds.time.values.item())
    plt.subplots_adjust(left=0.01, right=0.89, bottom=0.05, wspace=0.32)
    with fsspec.open(fig_filename, "wb") as fig_file:
        fig.savefig(fig_file, dpi=100)
    plt.close(fig)


def _non_zero(ds: xr.Dataset, variables: Sequence, tol=1e-12) -> bool:
    """Check whether any of variables are non-zero. Useful to ensure that
    movies of all zero-valued fields are not generated."""
    for variable in variables:
        if abs(ds[variable]).max() > tol:
            return True
    return False


def _movie_funcs():
    """Return mapping of movie name to movie-still creation function.
    
    Each function must have following signature:

        func(arg: MovieArg)

        where arg is a tuple of an xr.Dataset containing the data to be plotted
        and a path for where func should save the figure it generates.
    """
    return {
        "column_ML_wind_tendencies": {
            "plotting_function": _save_wind_tendency_fig,
            "required_variables": list(WIND_TENDENCY_PLOT_KWARGS.keys()),
        },
        "column_heating_moistening": {
            "plotting_function": _save_heating_moistening_fig,
            "required_variables": list(HEATING_MOISTENING_PLOT_KWARGS.keys()),
        },
    }


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Path to rundir")
    parser.add_argument("output", help="Output location for movie stills")
    parser.add_argument("--n_jobs", default=8, type=int, help="Number of workers.")
    parser.add_argument(
        "--n_timesteps",
        default=None,
        type=int,
        help="Number of timesteps for which stills are generated.",
    )
    parser.add_argument("--catalog", default=vcm.catalog.catalog_path)
    args = parser.parse_args()

    if vcm.cloud.get_protocol(args.output) == "file":
        os.makedirs(args.output, exist_ok=True)

    catalog = intake.open_catalog(args.catalog)
    verification = config.get_verification_entries("40day_may2020", catalog)

    prognostic, _, grid = load_diags.load_physics(
        args.url, verification["physics"], catalog
    )
    # crashed prognostic runs have bad grid vars, so use grid from catalog instead
    prognostic = (
        prognostic.drop_vars(GRID_VARS, errors="ignore")
        .drop_dims(INTERFACE_DIMS, errors="ignore")
        .merge(grid)
    )

    if args.n_timesteps:
        prognostic = prognostic.isel(time=slice(None, args.n_timesteps))
    T = prognostic.sizes["time"]

    for name, movie_spec in _movie_funcs().items():
        func = movie_spec["plotting_function"]
        required_variables = movie_spec["required_variables"]
        logger.info(f"Forcing load for required variables for {name} movie")
        movie_data = prognostic[GRID_VARS + required_variables].load()
        filename = os.path.join(args.output, name + FIG_SUFFIX)
        func_args = [(movie_data.isel(time=t), filename.format(t=t)) for t in range(T)]
        if _non_zero(movie_data, required_variables):
            logger.info(f"Saving {T} still images for {name} movie to {args.output}")
            with get_context("spawn").Pool(args.n_jobs) as p:
                p.map(func, func_args)
        else:
            logger.info(f"Skipping {name} movie since all plotted variables are zero")
