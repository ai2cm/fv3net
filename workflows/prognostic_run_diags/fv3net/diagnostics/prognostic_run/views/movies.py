import logging
import os
from multiprocessing import get_context
import subprocess
import tempfile
from typing import Mapping, Sequence, Tuple

import dask


import intake
import fsspec
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import fv3viz as viz
import vcm
import vcm.catalog

from fv3net.diagnostics.prognostic_run import config
import fv3net.diagnostics.prognostic_run.load_diagnostic_data as load_diags

dask.config.set(sheduler="single-threaded")
logger = logging.getLogger(__name__)


MovieArg = Tuple[xr.Dataset, str]
FIG_SUFFIX = "_%05d.png"

COORD_NAMES = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "x_interface",
    "coord_y_outer": "y_interface",
}

_COORD_VARS = {
    "lonb": ["y_interface", "x_interface", "tile"],
    "latb": ["y_interface", "x_interface", "tile"],
    "lon": ["y", "x", "tile"],
    "lat": ["y", "x", "tile"],
}

GRID_VARS = ["area", "lonb", "latb", "lon", "lat"]
INTERFACE_DIMS = ["x_interface", "y_interface"]

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
        if variable in ds and abs(ds[variable]).max() > tol:
            return True
    return False


def _movie_specs():
    """Return mapping of movie name to movie specification.

    Movie specification is a mapping with a "plotting_function" key and
    a "required_variables" key.
    
    Each plotting function must have following signature:

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


def _create_movie(name: str, spec: Mapping, ds: xr.Dataset, output: str, n_jobs: int):
    fs = vcm.cloud.get_fs(output)
    func = spec["plotting_function"]
    required_variables = spec["required_variables"]
    logger.info(f"Forcing load for required variables for {name} movie")
    data = ds[GRID_VARS + required_variables].load()
    T = data.sizes["time"]
    if _non_zero(data, required_variables):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Saving {T} still images for {name} movie to {tmpdir}")
            filename = os.path.join(tmpdir, name + FIG_SUFFIX)
            func_args = [(data.isel(time=t), filename % t) for t in range(T)]
            with get_context("spawn").Pool(n_jobs) as p:
                p.map(func, func_args)
            movie_path = _stitch_movie_stills(filename)
            fs.put(movie_path, os.path.join(output, f"{name}.mp4"))
    else:
        logger.info(f"Skipping {name} movie since all plotted variables are zero")


def _get_ffpmeg_args(input_: str, output: str) -> Sequence[str]:
    return [
        "ffmpeg",
        "-y",
        "-r",
        "15",
        "-i",
        input_,
        "-vf",
        "fps=15",
        "-pix_fmt",
        "yuv420p",
        "-s:v",
        "1920x1080",
        output,
    ]


def _stitch_movie_stills(input_path):
    output_path = input_path[: -len(FIG_SUFFIX)] + ".mp4"
    ffmpeg_args = _get_ffpmeg_args(input_path, output_path)
    subprocess.check_call(ffmpeg_args)
    return output_path


def register_parser(subparsers):
    parser = subparsers.add_parser("movie", help="Generate movies from prognostic run.")
    parser.add_argument("url", help="Path to rundir.")
    parser.add_argument("output", help="Output location for movies.")
    parser.add_argument("--n_jobs", default=8, type=int, help="Number of workers.")
    parser.add_argument(
        "--n_timesteps",
        default=None,
        type=int,
        help="Number of timesteps to use in movie. If not provided all times are used.",
    )
    parser.add_argument("--catalog", default=vcm.catalog.catalog_path)
    parser.set_defaults(func=main)


def main(args):
    logging.basicConfig(level=logging.INFO)

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

    for name, movie_spec in _movie_specs().items():
        _create_movie(name, movie_spec, prognostic, args.output, args.n_jobs)
