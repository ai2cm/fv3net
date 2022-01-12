# %%
import os
import xarray
import vcm.catalog
import vcm
import fv3viz
import matplotlib.pyplot as plt
import numpy as np
import wandb
from fv3fit.tensorboard import plot_to_image

# %%
wandb.init(
    entity="ai2cm",
    project="microphysics-emulation",
    job_type="figures",
    name="experiment/mask-latitude-figures",
)


def _get_image(fig=None):
    if fig is None:
        fig = plt.gcf()
    return wandb.Image(plot_to_image(fig))


# %%
url = "gs://vcm-ml-experiments/microphysics-emulation/2022-01-12/rnn-v1-shared-mask50lat-10d-v3-online"  # noqa
file = "state_after_timestep.zarr"

# %%
grid = vcm.catalog.catalog["grid/c48"].to_dask()
ds = xarray.open_zarr(os.path.join(url, file)).merge(grid)
ds["z"] = ds.z


# %% Histogram
bins = 10.0 ** np.arange(-15, 0, 0.25)
ds.cloud_water_mixing_ratio.isel(time=0).plot.hist(bins=bins, histtype="step")
ds.cloud_water_mixing_ratio.isel(time=-1).plot.hist(bins=bins, histtype="step")
plt.legend([ds.time[0].item(), ds.time[-1].item()])
plt.xscale("log")
plt.yscale("log")
plt.ylim(top=1e7)

wandb.log({"histogram": _get_image()})

# %%
fig = fv3viz.plot_cube(
    ds.isel(time=[0, -1], z=[20, 43]),
    "cloud_water_mixing_ratio",
    row="z",
    col="time",
    vmax=0.0005,
)[0]
fig.set_size_inches(10, 5)
wandb.log({"maps": _get_image()})
