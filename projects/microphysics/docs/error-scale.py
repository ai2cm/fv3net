# flake8: noqa
# %% [markdown]
# ## Scaling the loss function
#
# The microphysics emulator model offline skill is worst in regions with low surface temperature ([report](https://datapane.com/u/noah5/reports/aAMZD23/2021-12-14microphysics-piggy-back-spatial-skill-rnn/)). Can we weight these regions more heavily in the loss? # noqa
#
# This results suggest yes. We load in this offline piggy-backed dataset, and compute the skill of a column prediction conditioned on the surface temperature of each column. # noqa

# %%
import os

import fsspec
import fv3viz
import matplotlib.pyplot as plt
import numpy as np
import xarray
from cycler import cycler
from vcm.fv3.metadata import gfdl_to_standard

# colorblind friendly settings
wong_palette = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

plt.rcParams["axes.prop_cycle"] = cycler("color", wong_palette)


def open_run(base_url):
    url = os.path.join(base_url, "piggy.zarr")
    ds = xarray.open_zarr(url)
    ds = gfdl_to_standard(ds)

    url = os.path.join(base_url, "atmos_dt_atmos.zarr")
    grid = xarray.open_zarr(fsspec.get_mapper(url))
    grid = gfdl_to_standard(grid)
    return ds.merge(grid, join="inner")


def compute_score(truth, prediction, average):
    return 1 - average((truth - prediction) ** 2) / average(truth ** 2)


def compute_mse(truth, prediction, average):
    return average((truth - prediction) ** 2)


def lat_average(lat, x):
    return x.groupby_bins(lat, bins=10).mean()


# %%
base_url = (
    "gs://vcm-ml-experiments/microphysics-emulation/2021-12-13/"
    "rnn-v1-shared-6hr-v2-offline"
)
ds = open_run(base_url)

stacked = ds.isel(time=0).stack(sample=["x", "y", "tile"])

emu = stacked.tendency_of_cloud_water_due_to_zhao_carr_emulator
phys = stacked.tendency_of_cloud_water_due_to_zhao_carr_physics


# %% [markdown]
# The skill is poor for low latitude:

# %%
score_by_lat = compute_score(phys, emu, lambda x: lat_average(stacked.lat, x).mean("z"))
score_by_lat.plot()


# %% [markdown]
# The effect is even clearer for low surface temperatures. For the lowest temperatures the skill -1000!

# %%


def surface_temp_average(x):
    return x.groupby_bins(stacked.TMPlowest, 30).mean().mean("z")


score_by_temp = compute_score(phys, emu, surface_temp_average)
score_by_temp.plot()

# %% [markdown]
# Is this loss in skill coming from a change in magnitude of the prediction or

# %%
ss = compute_mse(phys, 0, surface_temp_average)
ss_preds = compute_mse(emu, 0, surface_temp_average)
sse = compute_mse(phys, emu, surface_temp_average)

fig = plt.figure(figsize=(8, 8))


gs = fig.add_gridspec(
    2,
    1,
    height_ratios=(2, 7),
    left=0.1,
    right=0.9,
    bottom=0.1,
    top=0.9,
    wspace=0.05,
    hspace=0.25,
)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histx.tick_params(axis="x", labelbottom=False)
ss.plot(label="sum of square targets", yscale="log", ax=ax)
ss_preds.plot(label="sum of square predictions", yscale="log", ax=ax)
sse.plot(label="sum of squares errors", ax=ax)

T = ss["TMPlowest_bins"]
T = np.vectorize(lambda T: T.mid)(T)

scale_factor = np.where(T < 260, (T - 260) / 5 - 16 * np.log(10), -16 * np.log(10))
scale_factor = np.exp(scale_factor)
ax.plot(T, scale_factor, label="scale_factor")

theory = np.exp(17 * T / 243) / 1e23
ax.plot(T, theory, label="slope of saturation vapor pressure 17/243")


stacked.TMPlowest.plot.hist(ax=ax_histx)
ax.legend()
ax.grid()
ax_histx.grid()

# %% [markdown]
# This plot shows that the sum of square errors (MSE) exceeds the target sum of squares when the surface temperature is below 250 K. this is very cold indeed...very few points have this cold of a surface. The magnitude of the sum of squares predictions and targets starts to diverge around T=260. These points are mostly on or next to Antarctica.
#
# We should also be able to use the simple temperature "scale_factor" to scale
# the loss value in a given column. This formula is given by
# $$ S(T) = \min(T-260, 0) / 5 - 16 / \log(10) $$
#

# %%
plotme = ds.isel(time=0)
title = "surface_temperature_less_than_260"
plotme[title] = plotme.TMPlowest < 260
fv3viz.plot_cube(plotme, title)
