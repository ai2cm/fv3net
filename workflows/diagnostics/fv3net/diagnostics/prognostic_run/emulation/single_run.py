#!/usr/bin/env python
import numpy as np
import xarray as xr
import vcm
import vcm.catalog
import vcm.fv3.metadata
import wandb
import fv3viz
import plotly.express as px

import matplotlib.pyplot as plt

from fv3fit.tensorboard import plot_to_image
from . import tendencies


import argparse

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
    return ds.assign(time=np.vectorize(vcm.cast_to_datetime)(ds.time))


def get_url_wandb(job, artifact: str):
    art = job.use_artifact(artifact + ":latest", type="prognostic-run")
    path = "fv3config.yml"
    url = art.get_path(path).ref
    return url[: -len("/" + path)]


@register_log
def plot_histogram_begin_end(ds):
    ds = ds.dropna("time")
    bins = 10.0 ** np.arange(-15, 0, 0.25)
    ds.cloud_water_mixing_ratio.isel(time=0).plot.hist(bins=bins, histtype="step")
    ds.cloud_water_mixing_ratio.isel(time=-1).plot.hist(bins=bins, histtype="step")
    plt.legend([ds.time[0].item(), ds.time[-1].item()])
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(top=1e7)
    return {"cloud_histogram": _get_image()}


@register_log
def plot_cloud_weighted_average(ds):
    ds = ds.dropna("time")
    vcm.weighted_average(ds.cloud_water_mixing_ratio, ds.area).plot()
    plt.title("Global average cloud water")
    return {"global_average_cloud": _get_image()}


@register_log
def plot_cloud_maps(ds):
    ds = ds.dropna("time")
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
    return {"skill": time_dependent_dataset(skills_3d(ds))}


@register_log
def skill_time_table(ds):
    return {"skill_time": time_dependent_dataset(skills_1d(ds))}


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


def skills_3d(ds):
    out = {}
    for field in ["cloud_water", "specific_humidity", "air_temperature"]:
        prediction = tendencies.total_tendency(ds, field, source="emulator")
        truth = tendencies.total_tendency(ds, field, source="physics")
        out[field] = skill_improvement(truth, prediction, ds.area)
    return xr.Dataset(out)


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


def column_integrated_skills(ds):
    return xr.Dataset(
        dict(
            cloud_water=skill_improvement_column(
                ds.tendency_of_cloud_water_due_to_zhao_carr_physics,
                ds.tendency_of_cloud_water_due_to_zhao_carr_emulator,
                ds.area,
            ),
            specific_humidity=skill_improvement_column(
                ds.tendency_of_specific_humidity_due_to_zhao_carr_physics,
                ds.tendency_of_specific_humidity_due_to_zhao_carr_emulator,
                ds.area,
            ),
            air_temperature=skill_improvement_column(
                ds.tendency_of_air_temperature_due_to_zhao_carr_physics,
                ds.tendency_of_air_temperature_due_to_zhao_carr_emulator,
                ds.area,
            ),
            surface_precipitation=skill_improvement_column(
                ds.surface_precipitation_due_to_zhao_carr_physics,
                ds.surface_precipitation_due_to_zhao_carr_emulator,
                ds.area,
            ),
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


def register_parser(subparsers) -> None:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "piggy",
        help="Log piggy backed metrics for prognostic run named TAG"
        "to weights and biases.",
    )
    parser.add_argument("tag", help="The unique tag used for the prognostic run.")

    parser.set_defaults(func=main)


def main(args):

    run_artifact_path = args.tag

    job = wandb.init(
        job_type="piggy-back", project="microphysics-emulation", entity="ai2cm",
    )

    url = get_url_wandb(job, run_artifact_path)
    wandb.config["run"] = url
    grid = vcm.catalog.catalog["grid/c48"].to_dask()
    piggy = xr.open_zarr(url + "/piggy.zarr")
    state = xr.open_zarr(url + "/state_after_timestep.zarr")

    ds = vcm.fv3.metadata.gfdl_to_standard(piggy).merge(grid).merge(state)

    summary_functions = [
        lambda ds: {
            f"column_skill/{key}": float(val)
            for key, val in column_integrated_skills(ds).items()
        }
    ]

    for func in log_functions:
        print(f"Running {func}")
        wandb.log(func(ds))

    for func in summary_functions:
        for key, val in func(ds).items():
            wandb.summary[key] = val
