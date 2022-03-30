#!/usr/bin/env python
import argparse
import logging
from functools import partial
from typing import Any, Callable, List

import dask.diagnostics
import fv3viz
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import vcm
import vcm.catalog
import vcm.fv3.metadata
import xarray as xr
from fv3fit.tensorboard import plot_to_image

import wandb
from fv3net.diagnostics.prognostic_run.load_run_data import (
    open_segmented_logs_as_strings,
)
from fv3net.diagnostics.prognostic_run.logs import parse_duration

from . import tendencies

logger = logging.getLogger(__name__)

SKILL_FIELDS = ["cloud_water", "specific_humidity", "air_temperature"]

WANDB_PROJECT = "microphysics-emulation"
WANDB_ENTITY = "ai2cm"

log_functions = []
summary_functions = []


def register_log(func):
    log_functions.append(func)
    return func


def register_summary(func):
    summary_functions.append(func)
    return func


def compute_summaries(ds):
    out = {}
    for func in summary_functions:
        for key, val in func(ds).items():
            out[key] = val

    return out


def _get_image(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.set_size_inches(6, 4)
    im = wandb.Image(plot_to_image(fig))
    plt.close(fig)
    return im


def _cast_time_to_datetime(ds):
    return ds.assign_coords(time=np.vectorize(vcm.cast_to_datetime)(ds.time))


def get_url_wandb(job, artifact: str):
    art = job.use_artifact(artifact + ":latest", type="prognostic-run")
    path = "fv3config.yml"
    url = art.get_path(path).ref
    return url[: -len("/" + path)]


@register_log
def plot_histogram_begin_end(ds):
    bins = 10.0 ** np.arange(-15, 0, 0.25)
    ds.cloud_water_mixing_ratio.isel(time=0).plot.hist(bins=bins, histtype="step")
    # include second to last timestep if is nan
    ds.cloud_water_mixing_ratio.isel(time=-2).plot.hist(bins=bins, histtype="step")
    plt.legend([ds.time[0].item(), ds.time[-1].item()])
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(top=1e7)
    return {"cloud_histogram": _get_image()}


def _plot_cloud_time_vs_z(cloud: xr.DataArray):
    cloud.plot(vmin=-2e-5, vmax=2e-5, cmap="RdBu_r", y="z", yincrease=True)


@register_log
def plot_cloud_weighted_average(ds):
    global_cloud = vcm.weighted_average(ds.cloud_water_mixing_ratio, ds.area)
    _plot_cloud_time_vs_z(global_cloud)
    plt.title("Global average cloud water")
    return {"global_average_cloud": _get_image()}


@register_log
def plot_cloud_maps(ds):
    ds = ds.assign(z=ds.z)
    fig = fv3viz.plot_cube(
        ds.isel(time=[0, -1], z=[20, 43]),
        "cloud_water_mixing_ratio",
        row="z",
        col="time",
        vmax=0.0005,
    )[0]
    fig.set_size_inches(10, 5)
    return {"cloud_maps": _get_image()}


@register_log
def skill_table(ds):
    fields = SKILL_FIELDS
    out = {}
    for name, transform in [
        ("total", tendencies.total_tendency),
        ("gscond", tendencies.gscond_tendency),
        ("precpd", tendencies.precpd_tendency),
    ]:
        skills = skills_3d(ds, fields=fields, transform=transform)
        # total tendency named skill for backwards compatibility reasons
        out["total" if name == "total" else name] = time_dependent_dataset(skills)

        for field in skills:
            plotme = _cast_time_to_datetime(skills[field])
            out[f"skill/time_vs_lev/{name}/{field}"] = px.imshow(
                plotme.transpose("z", "time"),
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
            )

    return out


@register_log
def skill_time_table(ds):
    return {"skill_time": time_dependent_dataset(skills_1d(ds))}


def summarize_column_skill(ds, prefix, tendency_func):
    return {
        f"{prefix}/{field}": float(
            column_integrated_skill(ds, partial(tendency_func, field=field))
        )
        for field in SKILL_FIELDS
    }


for name, tendency_func in [
    # total tendency named skill for backwards compatibility reasons
    ("column_skill", tendencies.total_tendency),
    ("column_skill/gscond", tendencies.gscond_tendency),
    ("column_skill/precpd", tendencies.precpd_tendency),
]:
    register_summary(
        partial(summarize_column_skill, prefix=name, tendency_func=tendency_func)
    )


@register_summary
def summarize_precip_skill(ds):
    return {
        "column_skill/surface_precipitation": float(
            column_integrated_skill(ds, tendencies.surface_precipitation)
        )
    }


def mse(x: xr.DataArray, y, area, dims=None):
    if dims is None:
        dims = set(area.dims)
    return vcm.weighted_average((x - y) ** 2, area, dims)


def skill_improvement(truth, pred, area):
    return 1 - mse(truth, pred, area) / mse(truth, 0, area)


def skill_improvement_column(truth, pred, area):
    return 1 - mse(truth, pred, area).mean() / mse(truth, 0, area).mean()


def plot_r2(r2):
    r2.drop("z").plot(yincrease=False, y="z", vmax=1)


def skills_3d(
    ds: xr.Dataset,
    fields: List[str],
    transform: Callable[[xr.Dataset, str, str], xr.DataArray],
):
    out = {}
    for field in fields:
        prediction = transform(ds, field, source="emulator")
        truth = transform(ds, field, source="physics")
        out[field] = skill_improvement(truth, prediction, ds.area)
    return xr.Dataset(out)


def column_integrated_skill(
    ds: xr.Dataset, transform: Callable[[xr.Dataset, str], xr.DataArray],
):
    prediction = transform(ds, source="emulator")
    truth = transform(ds, source="physics")
    return skill_improvement_column(truth, prediction, ds.area)


def skills_1d(ds):
    return xr.Dataset(
        dict(
            surface_precipitation=skill_improvement(
                ds.surface_precipitation_due_to_zhao_carr_physics,
                ds.surface_precipitation_due_to_zhao_carr_emulator,
                ds.area,
            )
        )
    )


def plot_cloud_skill_zonal(ds, field, time):
    emu = ds[f"tendency_of_{field}_due_to_zhao_carr_emulator"]
    phys = ds[f"tendency_of_{field}_due_to_zhao_carr_physics"]

    num = vcm.zonal_average_approximate(ds.lat, (emu - phys) ** 2)
    denom = vcm.zonal_average_approximate(ds.lat, phys ** 2)

    score = 1 - num / denom
    if isinstance(time, int):
        plotme = score.isel(time=time)
        time = plotme.time.item().isoformat()
    else:
        plotme = score.mean("time")

    return px.imshow(plotme, zmin=-1, zmax=1, color_continuous_scale="RdBu_r")


def time_dependent_dataset(skills):
    skills = _cast_time_to_datetime(skills)
    df = skills.to_dataframe().reset_index()
    df["time"] = df.time.apply(lambda x: x.isoformat())
    return wandb.Table(dataframe=df)


def log_lat_vs_p_skill(field):
    """Returns a function that will compute the lat vs pressure skill of a field"""

    def func(ds):
        return {
            f"lat_p_skill/{field}": wandb.Plotly(
                plot_cloud_skill_zonal(ds, field, "mean")
            )
        }

    return func


for field in ["cloud_water", "specific_humidity", "air_temperature"]:
    register_log(log_lat_vs_p_skill(field))


def get_duration_seconds(url) -> int:
    logs = open_segmented_logs_as_strings(url)
    return parse_duration(logs).total_seconds()


def register_parser(subparsers) -> None:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "piggy",
        help="Log piggy backed metrics for prognostic run named TAG"
        "to weights and biases.",
    )
    parser.add_argument("tag", help="The unique tag used for the prognostic run.")
    parser.set_defaults(func=main)


def get_prognostic_run_from_tag(tag: str) -> Any:
    api = wandb.Api()
    runs = api.runs(filters={"group": tag}, path=f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    prognostic_runs = []

    for run in runs:
        if run.job_type == "prognostic_run":
            prognostic_runs.append(run)
    (run,) = prognostic_runs
    return run


def get_rundir_from_prognostic_run(run: Any) -> str:
    return run.config["rundir"]


def open_zarr(url: str):
    cachedir = "/tmp/files"
    logger.info(f"Opening {url} with caching at {cachedir}.")
    return xr.open_zarr(
        "filecache::" + url, storage_options={"filecache": {"cache_storage": cachedir}}
    )


def open_rundir(url):
    grid = vcm.catalog.catalog["grid/c48"].to_dask().load()
    piggy = open_zarr(url + "/piggy.zarr")
    state = open_zarr(url + "/state_after_timestep.zarr")
    return vcm.fv3.metadata.gfdl_to_standard(piggy).merge(grid).merge(state)


def upload_diagnostics_for_rundir(url):
    wandb.config["run"] = url
    wandb.summary["duration_seconds"] = get_duration_seconds(url)

    ds = open_rundir(url)

    for func in log_functions:
        print(f"Running {func}")
        with dask.diagnostics.ProgressBar():
            wandb.log(func(ds))

    for key, val in compute_summaries(ds).items():
        wandb.summary[key] = val


def upload_diagnostics_for_tag(tag: str):
    run = wandb.init(
        job_type="piggy-back",
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=tag,
        tags=[tag],
        reinit=True,
    )
    with run:
        url = get_rundir_from_prognostic_run(get_prognostic_run_from_tag(tag))
        upload_diagnostics_for_rundir(url)


def main(args):
    return upload_diagnostics_for_tag(args.tag)
