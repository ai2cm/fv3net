import dataclasses
import logging
import os
from multiprocessing import get_context
import subprocess
import tempfile
from typing import Callable, Sequence, Tuple

import dask


import intake
import fsspec
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import fv3viz
import vcm

from fv3net.diagnostics.prognostic_run import derived_variables
from fv3net.diagnostics.prognostic_run.compute import (
    add_catalog_and_verification_arguments,
    get_verification,
)
import fv3net.diagnostics.prognostic_run.load_run_data as load_diags

dask.config.set(sheduler="single-threaded")
logger = logging.getLogger(__name__)


MovieArg = Tuple[xr.Dataset, str]
FIG_SUFFIX = "_%05d.png"

GRID_VARS = ["area", "lonb", "latb", "lon", "lat"]
INTERFACE_DIMS = ["x_interface", "y_interface"]

HEATING_MOISTENING_PLOT_KWARGS = {
    "column_integrated_pQ1": {"vmin": -600, "vmax": 600, "cmap": "RdBu_r"},
    "column_integrated_dQ1_or_nQ1": {"vmin": -300, "vmax": 300, "cmap": "RdBu_r"},
    "column_integrated_Q1": {"vmin": -600, "vmax": 600, "cmap": "RdBu_r"},
    "column_integrated_pQ2": {"vmin": -20, "vmax": 20, "cmap": "RdBu_r"},
    "column_integrated_dQ2_or_nQ2": {"vmin": -10, "vmax": 10, "cmap": "RdBu_r"},
    "column_integrated_Q2": {"vmin": -20, "vmax": 20, "cmap": "RdBu_r"},
}

WIND_TENDENCY_PLOT_KWARGS = {
    "column_int_dQu": {"vmin": -4, "vmax": 4, "cmap": "RdBu_r"},
    "column_int_dQv": {"vmin": -4, "vmax": 4, "cmap": "RdBu_r"},
}

WATER_VAPOR_PATH_PLOT_KWARGS = {
    "water_vapor_path": {"vmin": 0, "vmax": 60, "cmap": "viridis"},
    "water_vapor_path_verification": {"vmin": 0, "vmax": 60, "cmap": "viridis"},
    "water_vapor_path_bias": {"vmin": -10, "vmax": 10, "cmap": "RdBu_r"},
}


@dataclasses.dataclass
class MovieSpec:
    name: str
    plotting_function: Callable[[MovieArg], None]
    required_variables: Sequence[str]
    require_verification_data: bool = False


def _plot_maps(ds, axes, plot_kwargs):
    for i, (variable, variable_plot_kwargs) in enumerate(plot_kwargs.items()):
        ax = axes.flatten()[i]
        fv3viz.plot_cube(ds, variable, ax=ax, **variable_plot_kwargs)
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


def _save_water_vapor_path_fig(arg: MovieArg):
    ds, fig_filename = arg
    with xr.set_options(keep_attrs=True):
        ds["water_vapor_path_bias"] = (
            ds["water_vapor_path"] - ds["water_vapor_path_verification"]
        )
    print(f"Saving to {fig_filename}")
    fig, axes = plt.subplots(
        1, 3, figsize=(14, 2.8), subplot_kw={"projection": ccrs.Robinson()}
    )
    _plot_maps(ds, axes, WATER_VAPOR_PATH_PLOT_KWARGS)
    fig.suptitle(ds.time.values.item())
    plt.subplots_adjust(left=0.01, right=0.92, bottom=0.05, wspace=0.25)
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


_MOVIE_SPECS = [
    MovieSpec(
        "column_ML_wind_tendencies",
        _save_wind_tendency_fig,
        list(WIND_TENDENCY_PLOT_KWARGS.keys()),
    ),
    MovieSpec(
        "column_heating_moistening",
        _save_heating_moistening_fig,
        list(HEATING_MOISTENING_PLOT_KWARGS.keys()),
    ),
    MovieSpec(
        "water_vapor_path",
        _save_water_vapor_path_fig,
        ["water_vapor_path", "water_vapor_path_verification"],
        require_verification_data=True,
    ),
]


def _create_movie(spec: MovieSpec, ds: xr.Dataset, output: str, n_jobs: int):
    fs = vcm.cloud.get_fs(output)
    name = spec.name
    required_variables = list(spec.required_variables)
    logger.info(f"Forcing load for required variables for {name} movie")
    data = ds[GRID_VARS + required_variables].load()
    T = data.sizes["time"]
    if _non_zero(data, required_variables):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Saving {T} still images for {name} movie to {tmpdir}")
            filename = os.path.join(tmpdir, name + FIG_SUFFIX)
            func_args = [(data.isel(time=t), filename % t) for t in range(T)]
            with get_context("spawn").Pool(n_jobs) as p:
                p.map(spec.plotting_function, func_args)
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
    add_catalog_and_verification_arguments(parser)
    parser.set_defaults(func=main)


def _merge_prognostic_verification(
    prognostic: xr.Dataset, verification: xr.Dataset
) -> xr.Dataset:
    """Merge prognostic and verification datasets into a single dataset."""
    common_variables = set(prognostic.data_vars).intersection(verification.data_vars)
    verification_renamed = verification[common_variables].rename(
        {v: f"{v}_verification" for v in common_variables}
    )
    return xr.merge([prognostic[common_variables], verification_renamed], join="inner")


def _limit_time_length(ds: xr.Dataset, n_timesteps: int) -> xr.Dataset:
    """Limit the time dimension of a dataset to the first n_timesteps."""
    max_time = min(n_timesteps, ds.sizes["time"])
    return ds.isel(time=slice(None, max_time))


def main(args):
    logging.basicConfig(level=logging.INFO)

    if vcm.cloud.get_protocol(args.output) == "file":
        os.makedirs(args.output, exist_ok=True)

    catalog = intake.open_catalog(args.catalog)
    grid = load_diags.load_grid(catalog)
    prognostic = derived_variables.physics_variables(
        load_diags.SegmentedRun(args.url, catalog).data_2d
    )
    verification = derived_variables.physics_variables(
        get_verification(args, catalog, join_2d="inner").data_2d
    )
    # crashed prognostic runs have bad grid vars, so use grid from catalog instead
    prognostic = (
        prognostic.drop_vars(GRID_VARS, errors="ignore")
        .drop_dims(INTERFACE_DIMS, errors="ignore")
        .merge(grid)
    )
    merged = _merge_prognostic_verification(prognostic, verification).merge(grid)

    if args.n_timesteps:
        prognostic = _limit_time_length(prognostic, args.n_timesteps)
        merged = _limit_time_length(merged, args.n_timesteps)

    for movie_spec in _MOVIE_SPECS:
        dataset = merged if movie_spec.require_verification_data else prognostic
        _create_movie(movie_spec, dataset, args.output, args.n_jobs)
