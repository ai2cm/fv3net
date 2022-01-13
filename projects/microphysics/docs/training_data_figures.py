# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Training Data
#
# The training data is generated from several simulations with the FV3GFS
# atmospheric model

# %%

import subprocess

import matplotlib.pyplot as plt
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
# open data
config = TrainConfig.from_yaml_path(
    "gs://vcm-ml-experiments/microphysics-emulation/2022-01-04/"
    "log-cloud-dense-9b3e1a/config.yaml"
)

train_ds = nc_dir_to_tf_dataset(
    config.train_url, config.get_dataset_convertor(), nfiles=config.nfiles
)
train_set = next(iter(train_ds.batch(40000)))
train_set = config.get_transform().forward(train_set)


# %% [markdown]
# The following conditional averaged plots show that a residual model works well
# for specific humidity and air temperature, but much less so for cloud water
# mixing ratio ($q_c$).
# The cloud water mixing ratio likely depends strongly on the relative humidity
# tendency by the non-grid-scale condensation processes.

# %%
fields = ZhaoCarrFields()
qc = fields.cloud_water


def conditional(y, x, bins=100):
    """condtional pdf p(y|x)"""
    y = np.asarray(y).ravel()
    x = np.asarray(x).ravel()
    f, xe, ye = np.histogram2d(x, y, bins=bins)
    conditional_pdf = f / f.sum(0)

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    fig = plt.figure()
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=(7, 2),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.pcolormesh(xe, ye, conditional_pdf, cmap="bone_r")
    lim = [x.min(), x.max()]
    ax.plot(lim, lim, color="green", lw=2)

    ax_hist.hist(x, bins=bins)
    return ax, ax_hist


def conditional_field(train_set, qc: Field):
    ax, ax_hist = conditional(
        train_set[qc.output_name], train_set[qc.input_name], bins=30
    )
    ax_hist.set_xlabel(qc.input_name)
    ax.set_ylabel(qc.output_name)
    ax.set_title(f"{qc.output_name} | {qc.input_name}")


plt.figure()
conditional_field(train_set, fields.air_temperature)

plt.figure()
conditional_field(train_set, fields.cloud_water)

plt.figure()
conditional_field(train_set, fields.specific_humidity)

log_qc = Field("log_cloud_output", "log_cloud_input")
plt.figure()
conditional_field(train_set, log_qc)


# %%
def plot_log_cloud_before_after(train_set, log=True):

    if log:
        bins = 10.0 ** np.arange(-40, 10, 1)
        scale = "log"
    else:
        bins = 100
        scale = "linear"

    pred = train_set["cloud_water_mixing_ratio_after_precpd"].numpy()
    truth = train_set["cloud_water_mixing_ratio_input"].numpy()
    _, ax = plt.subplots()
    ax.hist(truth.ravel(), bins, histtype="step", label="before")
    ax.hist(pred.ravel(), bins, histtype="step", label="after")
    ax.set_ylabel("count")
    ax.set_xlabel(r"$q_c$ output (kg/kg)")
    ax.legend()

    ax.set_xscale(scale)
    return ax


plot_log_cloud_before_after(train_set)
glue("qc-before-after", plt.gcf())

after = train_set["cloud_water_mixing_ratio_after_precpd"].numpy()
before = train_set["cloud_water_mixing_ratio_input"].numpy()
delp = train_set["pressure_thickness_of_atmospheric_layer"].numpy()
before = np.mean(np.sum(delp * before / 9.81, -1))
after = np.mean(np.sum(delp * after / 9.81, -1))
total_change = after - before
glue("qc-before-after-total-percent", 100 * total_change / before)

# %%


def plot_max_by_temperature(train_set):
    df = pd.DataFrame(
        dict(
            temp=np.ravel(train_set["air_temperature_after_precpd"]),
            qc=np.ravel(train_set["cloud_water_mixing_ratio_after_precpd"]),
            qv=np.ravel(train_set["specific_humidity_after_precpd"]),
        )
    )
    bins = np.arange(170, 320, 2.5)
    max = (
        df.drop("temp", axis=1)
        .groupby(pd.cut(df.temp, bins))
        .quantile(0.999)
        .fillna(1e-7)
        .reset_index()
    )
    max["temp"] = max.temp.apply(lambda x: x.mid)
    return max


df = plot_max_by_temperature(train_set)
df.plot(
    x="temp",
    y="qc",
    ylabel="99.9%-tile of cloud-water (kg/kg)",
    xlabel="Temperature (K)",
)
temp = df.temp.tolist()
qc = df.qc.tolist()
print("# 99.9%-tile cloud conditioned on temperature")
print(f"# computed in {__file__}")
sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
print(f"# git-rev: {sha}")
print("temp = ", temp)
print("qc =", qc)
f = scipy.interpolate.interp1d(temp, qc, fill_value=1e-7, bounds_error=False)
# %%
df.plot(
    x="temp",
    y="qv",
    ylabel="99.9%-tile of humidity (kg/kg)",
    xlabel="Temperature (K)",
    logy=True,
)
