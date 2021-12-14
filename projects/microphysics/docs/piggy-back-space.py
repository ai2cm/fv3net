# %%
import os
import datapane
import plotly.express as px
import datetime

import fv3viz
import numpy as np
import vcm
import xarray
from vcm.fv3.metadata import gfdl_to_standard


# %%
def open_run(base_url):
    url = os.path.join(base_url, "piggy.zarr")
    ds = xarray.open_zarr(url)
    ds = gfdl_to_standard(ds)

    url = os.path.join(base_url, "atmos_dt_atmos.zarr")
    grid = xarray.open_zarr(url)
    grid = gfdl_to_standard(grid)
    return ds.merge(grid)


# %%
base_url = (
    "gs://vcm-ml-experiments/microphysics-emulation/2021-12-13/"
    "rnn-v1-shared-6hr-v2-online"
)
ds = open_run(base_url)
emu = ds.tendency_of_cloud_water_due_to_zhao_carr_emulator
phys = ds.tendency_of_cloud_water_due_to_zhao_carr_physics

score = xarray.where(phys ** 2 > 1e-15, 1 - (emu - phys) ** 2 / phys ** 2, np.nan)
plotme = ds.assign(score=score)
fv3viz.plot_cube(plotme.isel(z=70, time=-1), "score", vmin=-1, vmax=1)

# %%


def plot_cloud_skill_zonal(base_url, field, time):
    ds = open_run(base_url)
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

    fig = px.imshow(plotme, zmin=-1, zmax=1, color_continuous_scale="RdBu_r")

    return datapane.Plot(
        fig, caption=f"{field} tendency skill. Time: {time}. Url: {base_url}."
    )


# %%
groups = []
for field in ["cloud_water", "specific_humidity", "air_temperature"]:
    group = datapane.Group(
        datapane.Text("### Online"),
        plot_cloud_skill_zonal(
            (
                "gs://vcm-ml-experiments/microphysics-emulation/2021-12-13/"
                "rnn-v1-shared-6hr-v2-online"
            ),
            field,
            time="mean",
        ),
        datapane.Text("### Offline"),
        plot_cloud_skill_zonal(
            (
                "gs://vcm-ml-experiments/microphysics-emulation/2021-12-13/"
                "rnn-v1-shared-6hr-v2-offline"
            ),
            field,
            time="mean",
        ),
        label=field,
    )

    groups.append(group)

# %%
report = datapane.Report(datapane.Select(blocks=groups))
fmt = datetime.date.today().isoformat()
report.upload(name=f"{fmt}/microphysics-piggy-back-spatial-skill-rnn")
