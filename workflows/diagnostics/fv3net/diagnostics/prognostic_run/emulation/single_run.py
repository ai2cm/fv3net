#!/usr/bin/env python
import argparse
from datetime import timedelta
import logging
import os
from functools import partial
import re
import sys
from typing import Any, Callable, Iterable, List, Optional, Tuple
import cftime

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
from fv3net.artifacts.metadata import StepMetadata, log_fact_json

import wandb
from fv3net.diagnostics.prognostic_run.load_run_data import (
    open_segmented_logs_as_strings,
)
from fv3net.diagnostics.prognostic_run.logs import parse_duration
from fv3net.diagnostics.prognostic_run.emulation import query

from . import tendencies

logger = logging.getLogger(__name__)

SKILL_FIELDS = ["cloud_water", "specific_humidity", "air_temperature"]

WANDB_PROJECT = "microphysics-emulation"
WANDB_ENTITY = "ai2cm"
WANDB_JOB_TYPE = "piggy-back"

log_functions = []


def register_log(func):
    log_functions.append(func)
    return func


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
            column_integrated_skill(ds, lambda x, y: tendency_func(x, field, y))
        )
        for field in SKILL_FIELDS
    }.items()


def _global_average_cloud_ppm(ds, time, z) -> Optional[float]:
    field = "cloud_water_mixing_ratio"
    to_parts_per_million = 1e6

    try:
        selected = ds[field].sel(time=time)
    except KeyError:
        logger.warn("No field {} or time {}".format(field, time))
        return None

    selected_height = selected.interp(z=z)
    average_cloud = float(
        vcm.weighted_average(selected_height, ds.area, dims=selected_height.dims)
    )
    return average_cloud * to_parts_per_million


def global_average_cloud_5d_300mb_ppm(
    ds: xr.Dataset,
) -> Iterable[Tuple[str, Optional[float]]]:

    time = cftime.DatetimeJulian(2016, 6, 16)
    z = 300

    yield (
        global_average_cloud_5d_300mb_ppm.__name__,
        _global_average_cloud_ppm(ds, time, z),
    )


def global_average_cloud_1d_200mb_ppm(
    ds: xr.Dataset,
) -> Iterable[Tuple[str, Optional[float]]]:

    time = cftime.DatetimeJulian(2016, 7, 2)
    z = 200

    yield (
        "global_average_cloud_July2_200mb_ppm",
        _global_average_cloud_ppm(ds, time, z),
    )


def tropical_average_temp_near_tropopause_5d_k(
    ds: xr.Dataset,
) -> Iterable[Tuple[str, Optional[float]]]:

    func_name = "tropical_average_temp_near_tropopause_5d_k"
    field = "air_temperature"
    first_time = ds.time.isel(time=0).values
    time = first_time + timedelta(days=5)

    try:
        selected_time = ds[field].sel(time=time)
    except KeyError:
        logger.warn("No field {} or time {}".format(field, time))
        return (func_name, None)

    vert_average = selected_time.sel(z=slice(100, 200)).mean(dim="z")
    tropics_mask = np.logical_and(ds["lat"] < 30, ds["lat"] > -30)
    tropics_vert_average = vert_average.where(tropics_mask)
    average_temperature = float(
        vcm.weighted_average(
            tropics_vert_average, ds.area, dims=tropics_vert_average.dims
        )
    )
    yield (func_name, average_temperature)


def antarctic_avg_column_integrated_cloud_5d_kg(
    ds: xr.Dataset,
) -> Iterable[Tuple[str, Optional[float]]]:

    func_name = "antarctic_avg_column_integrated_cloud_5d_kg"
    field = "cloud_water_mixing_ratio"
    mass_field = "pressure_thickness_of_atmospheric_layer"
    first_time = ds.time.isel(time=0).values
    time = first_time + timedelta(days=5)

    integrated_cloud = vcm.mass_integrate(ds[field], ds[mass_field], dim="z")
    try:
        selected_time = integrated_cloud.sel(time=time)
    except KeyError:
        logger.warn("No field {} or time {}".format(field, time))
        return (func_name, None)

    antarctic_mask = ds["lat"] < -60
    average_cloud = float(
        vcm.weighted_average(
            selected_time.where(antarctic_mask), ds.area, dims=selected_time.dims
        )
    )
    yield (func_name, average_cloud)


def summarize_precip_skill(ds):
    yield "column_skill/surface_precipitation", float(
        column_integrated_skill(ds, tendencies.surface_precipitation)
    )


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
        prediction = transform(ds, field, "emulator")
        truth = transform(ds, field, "physics")
        out[field] = skill_improvement(truth, prediction, ds.area)
    return xr.Dataset(out)


def column_integrated_skill(
    ds: xr.Dataset, transform: Callable[[xr.Dataset, str], xr.DataArray],
):
    prediction = transform(ds, "emulator")
    truth = transform(ds, "physics")
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
    parser.add_argument(
        "--url",
        help="The path to the run-directory...in cases when tag alone is not enough "
        "like a crashed run.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-s", "--summary-only", help="Only run summaries.", action="store_true"
    )
    parser.add_argument(
        "--summary-filter", help="Regex to select summaries", type=str, default=".*"
    )
    parser.set_defaults(func=main)


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

    piggy = vcm.fv3.standardize_fv3_diagnostics(piggy)
    state = vcm.fv3.standardize_fv3_diagnostics(state)

    return piggy.merge(grid).merge(state)


def log_summary_wandb(summary):
    for key, val in summary.items():
        wandb.summary[key] = val


def upload_diagnostics_for_rundir(url: str, summary_only: bool, summary_name: str):
    summary_metrics = {"duration_seconds": get_duration_seconds(url)}
    print("duration_seconds", summary_metrics["duration_seconds"])

    wandb.config["run"] = url

    ds = open_rundir(url)

    if not summary_only:
        for func in log_functions:
            print(f"Running {func}")
            with dask.diagnostics.ProgressBar():
                wandb.log(func(ds))

    for func in get_summary_functions():
        for key, val in func(ds):
            if re.match(summary_name, key):
                print(key, val)
                summary_metrics[key] = val

    log_summary_wandb(summary_metrics)
    log_fact_json(data=summary_metrics)


def get_summary_functions() -> Iterable[
    Callable[[xr.Dataset], Iterable[Tuple[str, Any]]]
]:

    # build list of summaries
    yield global_average_cloud_5d_300mb_ppm
    yield global_average_cloud_1d_200mb_ppm
    yield summarize_precip_skill
    yield antarctic_avg_column_integrated_cloud_5d_kg
    yield tropical_average_temp_near_tropopause_5d_k

    for name, tendency_func in [
        # total tendency named skill for backwards compatibility reasons
        ("column_skill", tendencies.total_tendency),
        ("column_skill/gscond", tendencies.gscond_tendency),
        ("column_skill/precpd", tendencies.precpd_tendency),
    ]:
        func: Callable = partial(
            summarize_column_skill, prefix=name, tendency_func=tendency_func
        )
        func.__name__ = name
        yield func


def upload_diagnostics_for_tag(
    tag: str, summary_only: bool, summary_name: str, url: str = ""
):
    run = wandb.init(
        job_type=WANDB_JOB_TYPE,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=tag,
        tags=[tag],
        reinit=True,
    )

    if not url:
        api = wandb.Api()
        prognostic_run = query.PrognosticRunClient(
            tag, entity=run.entity, project=run.project, api=api
        )
        prognostic_run.use_artifact_in(run)
        url = prognostic_run.get_rundir_url()

    wandb.config["env"] = {"COMMIT_SHA": os.getenv("COMMIT_SHA", "")}
    StepMetadata(
        job_type=WANDB_JOB_TYPE,
        url=url,
        dependencies={"prognostic_run": url},
        args=sys.argv[1:],
    ).print_json()

    with run:
        upload_diagnostics_for_rundir(url, summary_only, summary_name)


def main(args):
    return upload_diagnostics_for_tag(
        args.tag,
        summary_only=args.summary_only,
        summary_name=args.summary_filter,
        url=args.url,
    )
