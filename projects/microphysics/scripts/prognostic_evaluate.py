import argparse
import os
from typing import Mapping, Optional
import logging
import hashlib
import tempfile
import wandb
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
from dask.diagnostics import ProgressBar
from wandb.errors import CommError


from fv3viz import infer_cmap_params, plot_cube
from fv3fit.tensorboard import plot_to_image
from vcm import interpolate_unstructured
from vcm.select import meridional_ring, zonal_average_approximate
from vcm.catalog import catalog


logger = logging.getLogger(__name__)


COMPARE_VARS = [
    "cloud_water_mixing_ratio",
    "specific_humidity",
    "air_temperature",
    "eastward_wind",
    "northward_wind",
]
TRANSECT_VARS = [
    "cloud_water_mixing_ratio",
    "specific_humidity",
]


def consistent_time_len(*da_args):

    # assumes same start time and time delta
    times = [len(da.time) for da in da_args]
    min_len = np.min(times)
    return [da.isel(time=slice(0, min_len)) for da in da_args]


# TODO: test this and maybe port back to diagnostics?
def _to_weighted_mean(da: xr.DataArray, area: xr.DataArray, dim: str):
    return da.weighted(area).mean(dim=dim)


def new_weighted_avg(ds: xr.Dataset, dim=["tile", "y", "x"]):

    area = ds["area"]
    area = area.fillna(0)
    weighted_mean = ds.map(_to_weighted_mean, keep_attrs=True, args=(area, dim))

    return weighted_mean


def get_avg_data(
    source_path: str,
    grid_data: xr.Dataset,
    run,
    filename="state_mean_by_height.nc",
    override_artifact=False,
):
    """
    Open a dataset and get the tile, x, y averaged data. Loads from
    a saved artifact based on the source_path, creates an artifact
    if it doesn't exist, or updates the artifact if an override is
    specified.

    Args:
        source_path: path to the zarr dataset we want to average
        grid_data: loaded grid information for simulation
        run: current wandb.init run
        filename: filename to store the averaged data at in the artifact
        override_artifact: ignore saved artifact and re-average data from source
    """

    source_hash = hashlib.md5(source_path.encode("utf-8")).hexdigest()
    try:
        artifact = run.use_artifact(f"{source_hash}:latest")
        logger.info(f"Loaded existing artifact for: {source_path}")
    except CommError:
        logger.error(f"Run averaging artifact not found for: {source_path}")
        artifact = None

    if override_artifact or artifact is None:
        # open zarr and do the average
        ds = xr.open_zarr(source_path, consolidated=True,).merge(grid_data)
        prog_avg = new_weighted_avg(ds)
        with ProgressBar():
            prog_avg.load()

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_file = os.path.join(tmpdir, filename)
            prog_avg.to_netcdf(nc_file)
            artifact = wandb.Artifact(source_hash, type="by_height_avg")
            artifact.add_file(nc_file)
            run.log_artifact(artifact)
    else:
        artifact_dir = artifact.download()
        prog_avg = xr.open_dataset(os.path.join(artifact_dir, filename))

    return prog_avg


def avg_vertical(ds: xr.Dataset, z_dim_names=["z", "z_soil"]):

    dims_to_avg = [dim for dim in z_dim_names if dim in ds.dims]
    return ds.mean(dim=dims_to_avg).compute()


def plot_global_avg_by_height_panel(
    da1: xr.DataArray, da2: xr.DataArray, x="time", dpi=80
):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(12, 4)
    fig.set_dpi(dpi)

    vmin, vmax, cmap = infer_cmap_params(da2, robust=True)
    vkw = dict(vmin=vmin, vmax=vmax, cmap=cmap)
    da1.plot.pcolormesh(x=x, y="z", ax=ax[0], yincrease=False, **vkw)
    da2.plot.pcolormesh(x=x, y="z", ax=ax[1], yincrease=False, **vkw)
    (da1 - da2).plot.pcolormesh(x=x, y="z", ax=ax[2], yincrease=False)
    ax[0].set_title("Emulation")
    ax[1].set_title("Baseline")
    ax[2].set_title("Diff")

    for sub_ax in ax:
        sub_ax.tick_params(axis="x", labelrotation=15)

    plt.tight_layout()

    return fig


def plot_time_heights(
    prognostic: xr.Dataset, baseline: xr.Dataset, do_variables=COMPARE_VARS
):

    prognostic, baseline = consistent_time_len(prognostic, baseline)

    for name in do_variables:
        fig = plot_global_avg_by_height_panel(prognostic[name], baseline[name])
        wandb.log({f"avg_time_height/{name}": wandb.Image(plot_to_image(fig))})
        plt.close(fig)


def plot_lat_heights(
    prognostic: xr.Dataset, baseline: xr.Dataset, do_variables=COMPARE_VARS
):

    prognostic, baseline = consistent_time_len(prognostic, baseline)
    ntimes = len(prognostic.time)
    start = max(ntimes - 8, 0)
    selection = slice(start, ntimes)
    prog_near_end = prognostic.isel(time=selection).mean(dim="time")
    base_near_end = baseline.isel(time=selection).mean(dim="time")
    lat = base_near_end["lat"]

    for name in do_variables:
        prog_zonal = zonal_average_approximate(lat, prog_near_end[name])
        base_zonal = zonal_average_approximate(lat, base_near_end[name])

        fig = plot_global_avg_by_height_panel(prog_zonal, base_zonal, x="lat")
        wandb.log({f"zonal_avg/{name}": wandb.Image(plot_to_image(fig))})
        plt.close(fig)


def plot_global_means(
    prognostic: xr.Dataset, baseline: xr.Dataset, do_variables=COMPARE_VARS
):

    prognostic, baseline = consistent_time_len(prognostic, baseline)

    for name in do_variables:
        if name not in baseline or name not in prognostic:
            logger.info(f"Skipping global mean due to missing variable: {name}")
            continue

        fig, ax = plt.subplots()
        fig.set_dpi(80)

        da = prognostic[name]
        da.plot(ax=ax, label="Emulation")
        baseline[name].plot(ax=ax, label="Baseline", alpha=0.6)
        plt.legend()

        wandb.log({f"global_avg/{name}": wandb.Image(plot_to_image(fig))})
        plt.close(fig)


def meridional_transect(ds: xr.Dataset):
    transect_coords = meridional_ring()
    return interpolate_unstructured(ds, transect_coords)


def plot_meridional(ds: xr.Dataset, varname: str, title="", ax=None, yincrease=False):
    meridional = meridional_transect(ds)
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_dpi(120)

    if "z_soil" in meridional[varname].dims:
        y = "z_soil"
    else:
        y = "z"

    vmin, vmax, cmap = infer_cmap_params(ds[varname], robust=True)
    if cmap == "viridis":
        cmap = "Blues"
    meridional[varname].plot.pcolormesh(
        x="lat", y=y, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, yincrease=yincrease,
    )
    ax.set_title(title, size=14)
    ax.set_ylabel("vertical level", size=12)
    ax.set_xlabel("latitude", size=12)


def plot_transects(
    prognostic: xr.Dataset, baseline: xr.Dataset, do_variables=TRANSECT_VARS
):

    tidx_map = {"start": 0, "near_end": len(prognostic.time) - 2}

    for time_name, tidx in tidx_map.items():
        for name in do_variables:
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(8, 4)
            fig.set_dpi(80)

            plot_meridional(prognostic.isel(time=tidx), name, ax=ax[0])
            plot_meridional(baseline.isel(time=tidx), name, ax=ax[1])

            ax[0].set_title("Emulation")
            ax[1].set_title("Baseline")
            plt.tight_layout()

            log_name = f"meridional_transect/{time_name}/{name}"
            wandb.log({log_name: wandb.Image(plot_to_image(fig))})
            plt.close(fig)


def plot_spatial_2panel_with_diff(emu: xr.Dataset, base: xr.Dataset, varname: str):

    fig = plt.figure()
    ax = fig.add_subplot(131, projection=ccrs.Robinson())
    ax2 = fig.add_subplot(132, projection=ccrs.Robinson())
    ax3 = fig.add_subplot(133, projection=ccrs.Robinson())
    fig.set_size_inches(15, 4)
    fig.set_dpi(80)

    vmin, vmax, cmap = infer_cmap_params(emu[varname], robust=True)
    plot_kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap)

    # TODO: can't pass explicit vmin, vmax so cbars arent equivalent
    plot_cube(emu, varname, ax=ax, cmap_percentiles_lim=False, **plot_kwargs)
    plot_cube(base, varname, ax=ax2, cmap_percentiles_lim=False, **plot_kwargs)

    diff = emu.copy()
    # Insert diff to keep undiffed grid coordinates
    diff[varname] = emu[varname] - base[varname]
    plot_cube(diff, varname, ax=ax3)

    ax.set_title(f"Emulation: {varname}")
    ax2.set_title("Baseline")
    ax3.set_title("Diff: Emu - Baseline")

    return fig


def plot_spatial_comparisons(
    prognostic: xr.Dataset,
    baseline: xr.Dataset,
    do_variables=COMPARE_VARS,
    time_idxs: Mapping[str, int] = None,
    level_map: Mapping[str, int] = None,
):

    if time_idxs is None:
        time_idxs = {"start": 0, "near_end": len(prognostic.time) - 2}

    if level_map is None:
        level_map = {"lower": 75, "upper_BL": 60, "upper_atm": 20}

    for name in do_variables:
        for level, lev_idx in level_map.items():
            for time, tidx in time_idxs.items():
                prog = prognostic.isel(time=tidx, z=lev_idx)
                base = baseline.isel(time=tidx, z=lev_idx)
                fig = plot_spatial_2panel_with_diff(prog, base, name)

                log_name = f"spatial_comparison/{time}/{level}/{name}"
                wandb.log({log_name: wandb.Image(plot_to_image(fig))})
                plt.close(fig)


def _selection(
    times: xr.DataArray, duration: timedelta, window: Optional[timedelta] = None
):

    """
    Generate a slice window at the specified duration inferred
    from the save timestep frequency

    Args:
        times: Datarray of datetime objects marking the savepoints
        duration: time elapsed to center the selection window at
        window: width of the selection window, defaults to +/- one timestep
    """

    delta = pd.Timedelta((times[1] - times[0]).values)
    center = duration // delta

    if window is None:
        increment = 1
    else:
        increment = window // delta

    start = max(center - increment, 0)
    end = center + increment

    return slice(start, end)


def log_all_drifts(
    prog_global_avg: xr.Dataset, base_global_avg: xr.Dataset, do_variables=COMPARE_VARS
):
    """
    Stores drift from baseline at different run lengths.
    All values stored in <units> / day
    """

    times = prog_global_avg.time
    for name in do_variables:
        # assumes time delta
        drift_sel = [(3,)]
        drift_sel = {
            "3hr": (timedelta(hours=3), None),
            "1day": (timedelta(days=1), timedelta(hours=12)),
            "5day": (timedelta(days=5), timedelta(hours=12)),
            "10day": (timedelta(days=10), timedelta(days=1)),
        }

        for key, (duration, window) in drift_sel.items():

            selection = _selection(times, duration, window=window)
            prog_sel = prog_global_avg[name].isel(time=selection).mean(dim="time")
            base_sel = base_global_avg[name].isel(time=selection).mean(dim="time")

            duration_in_days = (
                duration.total_seconds() / timedelta(days=1).total_seconds()
            )
            value = (prog_sel - base_sel).values.item() / duration_in_days

            wandb.log({f"drifts/{name}/{key}": value})


def main():

    parser = argparse.ArgumentParser(
        "Prognostic evaluation metrics",
        description="Log evaluation metrics for prognostic run named TAG",
    )
    parser.add_argument("tag", help="The unique tag used for the prognostic run.")
    parser.add_argument("--baseline-tag", default="baseline", help="The unique tag used for the baseline comparison run.")

    parser.add_argument(
        "--grid-key",
        default="c48",
        help="Grid to load from catalog for area-weighted averages",
    )
    parser.add_argument(
        "--override-artifacts",
        action="store_true",
        help="Force upload of the averaging artifacts",
    )
    parser.add_argument(
        "--wandb-project", default="microphysics-emulation",
    )

    args = parser.parse_args()
    run = wandb.init(
        job_type="prognostic_evaluation", entity="ai2cm", project=args.wandb_project
    )

    wandb.config.update(args)

    def get_url_wandb(artifact: str):
        art = job.use_artifact(artifact + ":latest", type="prognostic-run")
        path = "fv3config.yml"
        url = art.get_path(path).ref
        return url[: -len("/" + path)]

    prog_url = get_url_wandb(args.tag)
    baseline_url = get_url_wandb(args.baseline_tag)
    wandb.config["run"] = prog_url
    wandb.config["baseline_run"] = baseline_url

    prog = xr.open_zarr(os.path.join(prog_url, "state_after_timestep.zarr"), consolidated=True)
    baseline = xr.open_zarr(os.path.join(baseline_url, "state_after_timestep.zarr"), consolidated=True)
    grid = catalog[f"grid/{args.grid_key}"].to_dask()
    prog = prog.merge(grid)
    baseline = baseline.merge(grid)

    prog_mean_by_height = get_avg_data(
        path, grid, run, override_artifact=args.override_artifacts
    )
    base_mean_by_height = get_avg_data(
        baseline_path, grid, run, override_artifact=args.override_artifacts
    )

    plot_time_heights(prog_mean_by_height, base_mean_by_height)

    # Global average comparison after timestep
    prog_mean = avg_vertical(prog_mean_by_height)
    base_mean = avg_vertical(base_mean_by_height)

    plot_global_means(prog_mean, base_mean)
    log_all_drifts(prog_mean, base_mean)

    # Some meridional transects
    plot_transects(prog, baseline)

    # spatial plots
    plot_spatial_comparisons(prog, baseline)

    # zonal averages near the end (mean over 8 saved times)
    plot_lat_heights(prog, baseline)


if __name__ == "__main__":

    main()
