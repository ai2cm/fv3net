import argparse
import logging
import os
from functools import partial
from pathlib import Path
from multiprocessing import Pool
from typing import Callable
from toolz import curry

import intake
import fsspec
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

import vcm
import load_diagnostic_data as load_diags


_MOVIE_FUNCS = {}
MovieArg = (xr.Dataset, int, str)

HEATING_MOISTENING_PLOT_KWARGS = {
    "column_integrated_pQ1": {"vmin": -600, "vmax": 600, "cmap": "RdBu_r"},
    "column_integrated_dQ1": {"vmin": -600, "vmax": 600, "cmap": "RdBu_r"},
    "column_integrated_Q1": {"vmin": -600, "vmax": 600, "cmap": "RdBu_r"},
    "column_integrated_pQ2": {"vmin": -20, "vmax": 20, "cmap": "RdBu_r"},
    "column_integrated_dQ2": {"vmin": -20, "vmax": 20, "cmap": "RdBu_r"},
    "column_integrated_Q2": {"vmin": -20, "vmax": 20, "cmap": "RdBu_r"},
}

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

SUBPLOT_KW = {"projection": ccrs.Robinson()}


def _catalog():
    TOP_LEVEL_DIR = Path(os.path.abspath(__file__)).parent.parent.parent
    return str(TOP_LEVEL_DIR / "catalog.yml")


def _six_panel_heating_moistening(ds, axes):
    for i, (var, plot_kwargs) in enumerate(HEATING_MOISTENING_PLOT_KWARGS.items()):
        ax = axes.flatten()[i]
        mv = vcm.mappable_var(ds, var, coord_vars=_COORD_VARS, **COORD_NAMES)
        vcm.plot_cube(mv, ax=ax, **plot_kwargs)
        ax.set_title(var.replace("_", " "))


@curry
def add_to_movies(func: Callable[[int, xr.Dataset, str], None]):
    """Add a function to a series of movies to be created.

    Args:
        name: short description of movie. Will be used in filename.
        func: a function which creates and saves a single png to disk.
    """
    _MOVIE_FUNCS[func.__name__] = func


@add_to_movies
def column_heating_moistening(t, ds, filename_prefix):
    plotme = ds.isel(time=t)
    fig_filename = f"{filename_prefix}_{t:05}.png"
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.3), subplot_kw=SUBPLOT_KW)
    _six_panel_heating_moistening(plotme, axes)
    fig.suptitle(plotme.time.values.item())
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    with fsspec.open(fig_filename, "wb") as fig_file:
        fig.savefig(fig_file, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    CATALOG = _catalog()

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Path to rundir")
    parser.add_argument("grid_spec", help="Path to C384 grid spec (unused)")
    parser.add_argument("output", help="Output location for movie stills")
    args = parser.parse_args()

    if vcm.cloud.get_protocol(args.output) == "file":
        os.makedirs(args.output, exist_ok=True)

    catalog = intake.open_catalog(CATALOG)

    prognostic, _, grid = load_diags.load_physics(args.url, args.grid_spec, catalog)
    plot_vars = prognostic[list(HEATING_MOISTENING_PLOT_KWARGS.keys())]
    plot_vars = plot_vars.merge(grid)
    T = plot_vars.sizes["time"]
    for name, func in _MOVIE_FUNCS.items():
        logger.info(f"Saving {T} still images for {name} movie to {args.output}")
        prefix = os.path.join(args.output, name)
        with Pool(8) as p:
            p.map(partial(func, ds=plot_vars, filename_prefix=prefix), range(T))
