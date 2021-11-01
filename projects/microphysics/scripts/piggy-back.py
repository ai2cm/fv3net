#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import xarray as xr
import vcm
import vcm.catalog
import vcm.fv3.metadata
import wandb
import fv3viz

import matplotlib.pyplot as plt

from fv3fit.tensorboard import plot_to_image


run_artifact_path = sys.argv[1]

job = wandb.init(
    job_type="piggy-back", project="microphysics-emulation", entity="ai2cm"
)


def get_url_wandb(artifact: str):
    art = job.use_artifact(artifact)
    path = "fv3config.yml"
    url = art.get_path(path).ref
    return url[: -len("/" + path)]


url = get_url_wandb(run_artifact_path)
wandb.config["run"] = url
grid = vcm.catalog.catalog["grid/c48"].to_dask()
piggy = xr.open_zarr(url + "/piggy.zarr")

ds = vcm.fv3.metadata.gfdl_to_standard(piggy).merge(grid)
ds["time"] = np.vectorize(vcm.cast_to_datetime)(ds.time)

metrics = {}

for field in ["cloud_water", "specific_humidity", "air_temperature"]:
    emulator_var = f"tendency_of_{field}_due_to_zhao_carr_emulator"
    physics_var = f"tendency_of_{field}_due_to_zhao_carr_physics"

    fv3viz.plot_cube(ds.isel(time=0, z=50), emulator_var)
    metrics[emulator_var] = wandb.Image(plot_to_image(plt.gcf()))

    fv3viz.plot_cube(ds.isel(time=0, z=50), physics_var)
    metrics[physics_var] = wandb.Image(plot_to_image(plt.gcf()))
    plt.close("all")


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


skills = xr.Dataset(
    dict(
        cloud_water=skill_improvement(
            ds.tendency_of_cloud_water_due_to_zhao_carr_physics,
            ds.tendency_of_cloud_water_due_to_zhao_carr_emulator,
            ds.area,
        ),
        specific_humidity=skill_improvement(
            ds.tendency_of_specific_humidity_due_to_zhao_carr_physics,
            ds.tendency_of_specific_humidity_due_to_zhao_carr_emulator,
            ds.area,
        ),
        air_temperature=skill_improvement(
            ds.tendency_of_air_temperature_due_to_zhao_carr_physics,
            ds.tendency_of_air_temperature_due_to_zhao_carr_emulator,
            ds.area,
        ),
    )
)

skills_all = xr.Dataset(
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
    )
)

df = skills.to_dataframe().reset_index()
df["time"] = df.time.apply(lambda x: x.isoformat())
metrics["skill"] = wandb.Table(dataframe=df)
wandb.log(metrics)

for v in skills_all:
    wandb.summary["column_skill/" + v] = float(skills_all[v])
