# ---
# flake8: noqa
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: fv3net
#     language: python
#     name: fv3net
# ---

# %% [markdown]
# # Offline ML
# The training data is generated from several simulations with the FV3GFS
# atmospheric model

# %%

import subprocess

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import scipy.interpolate
from cycler import cycler
from fv3fit.emulation.zhao_carr_fields import Field, ZhaoCarrFields
from fv3fit.train_microphysics import TrainConfig, nc_dir_to_tf_dataset
from myst_nb import glue

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

# %%
# params
path = "gs://vcm-ml-experiments/microphysics-emulation/2022-01-18/rnn-predict-gscond-3f77ec"


# %%
# open data

config = TrainConfig.from_yaml_path(path + "/config.yaml")

train_ds = nc_dir_to_tf_dataset(
    config.train_url, config.get_dataset_convertor(), nfiles=config.nfiles
)
train_set = next(iter(train_ds.batch(40000)))
train_set = config.get_transform().forward(train_set)


# %%
import tensorflow as tf

model = tf.keras.models.load_model(path + "/model.tf")

# %%
predictions = model(train_set)


analysis = train_set.copy()
for key in predictions:
    analysis["predicted_" + key] = predictions[key]

# %%
df = pd.DataFrame(
    {key: np.ravel(array) for key, array in analysis.items() if array.shape[1] != 1}
)


for field in ["air_temperature", "specific_humidity", "cloud_water_mixing_ratio"]:
    df[f"{field}_persistence_error"] = (
        df[f"{field}_after_precpd"] - df[f"{field}_input"]
    )
    df[f"{field}_error"] = (
        df[f"{field}_after_precpd"] - df[f"predicted_{field}_after_precpd"]
    )


# %%
t_bins = np.arange(170, 320, 2.5)
membership = pd.cut(
    df["air_temperature_input"], t_bins, labels=(t_bins[1:] + t_bins[:-1]) / 2
)


def mean_square(x):
    return np.mean(x ** 2)


mean_square_df = df.groupby(membership).aggregate(mean_square)

# %%
fig, axs = plt.subplots(
    4,
    1,
    figsize=(4.77, 4.77),
    gridspec_kw=dict(height_ratios=[0.7, 2, 2, 2]),
    sharex=True,
    constrained_layout=True,
)

ax_hist = axs[0]
ax_hist.hist(df["air_temperature_input"], t_bins)
ax_hist.set_title("Histogram")

k = 1


def plot_field(field, ax, units=""):
    ax.plot(mean_square_df.index, mean_square_df[f"{field}_error"], label="error")
    ax.plot(
        mean_square_df.index,
        mean_square_df[f"{field}_persistence_error"],
        label="persistence_error",
    )
    ax.set_yscale("log")
    ax.grid()
    ax.set_title(field + " mean squared")
    ax.set_ylabel(units)
    ax.legend()


plot_field("cloud_water_mixing_ratio", axs[1], "(kg/kg)^2")
axs[1].set_ylim(bottom=1e-15)

plot_field("air_temperature", axs[2], "K^2")
plot_field("specific_humidity", axs[3], "(kg/kg)^2")

axs[-1].set_xlabel("Temperature (K)")

plt.savefig("error-scale.pdf")
