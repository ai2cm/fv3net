import argparse
import logging
import os
from pathlib import Path
from multiprocessing import get_context
from typing import Tuple

import dask


import intake
import fsspec
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import fv3viz as viz
import vcm
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

KEEP_VARS = GRID_VARS + list(HEATING_MOISTENING_PLOT_KWARGS.keys())


def _catalog():
    TOP_LEVEL_DIR = Path(os.path.abspath(__file__)).parent.parent.parent
    return str(TOP_LEVEL_DIR / "catalog.yml")


def _six_panel_heating_moistening(ds, axes):
    for i, (var, plot_kwargs) in enumerate(HEATING_MOISTENING_PLOT_KWARGS.items()):
        ax = axes.flatten()[i]
        mv = viz.mappable_var(ds, var, coord_vars=_COORD_VARS, **COORD_NAMES)
        viz.plot_cube(mv, ax=ax, **plot_kwargs)
        ax.set_title(var.replace("_", " "))


def _save_heating_moistening_fig(arg: MovieArg):
    ds, fig_filename = arg
    print(f"Saving to {fig_filename}")
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 5.3), subplot_kw={"projection": ccrs.Robinson()}
    )
    _six_panel_heating_moistening(ds, axes)
    fig.suptitle(ds.time.values.item())
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    with fsspec.open(fig_filename, "wb") as fig_file:
        fig.savefig(fig_file, dpi=100)
    plt.close(fig)


def _movie_funcs():
    """Return mapping of movie name to movie-still creation function.
    
    Each function must have following signature:

        func(arg: MovieArg)

        where arg is a tuple of an xr.Dataset containing the data to be plotted
        and a path for where func should save the figure it generates.
    """
    return {"column_heating_moistening": _save_heating_moistening_fig}


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    CATALOG = _catalog()

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Path to rundir")
    parser.add_argument("output", help="Output location for movie stills")
    args = parser.parse_args()

    if vcm.cloud.get_protocol(args.output) == "file":
        os.makedirs(args.output, exist_ok=True)

    catalog = intake.open_catalog(CATALOG)

    prognostic, _, grid = load_diags.load_physics(args.url, catalog)
    # crashed prognostic runs have bad grid vars, so use grid from catalog instead
    prognostic = (
        prognostic.drop_vars(GRID_VARS, errors="ignore")
        .drop_dims(INTERFACE_DIMS, errors="ignore")
        .merge(grid)
    )
    logger.info("Forcing computation")
    prognostic = prognostic[KEEP_VARS].load()  # force load
    T = prognostic.sizes["time"]
    for name, func in _movie_funcs().items():
        logger.info(f"Saving {T} still images for {name} movie to {args.output}")
        filename = os.path.join(args.output, name + FIG_SUFFIX)
        func_args = [(prognostic.isel(time=t), filename.format(t=t)) for t in range(T)]
        with get_context("spawn").Pool(8) as p:
            p.map(func, func_args)
