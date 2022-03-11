# flake8: noqa
# %%
from collections import defaultdict
from fv3net.diagnostics.prognostic_run.emulation.single_run import *
from fv3net.diagnostics.prognostic_run.emulation import tendencies
import numpy
import numpy as np
import vcm
import xarray as xr
import matplotlib.pyplot as plt
from cycler import cycler
import report
from fv3net.diagnostics.prognostic_run.views.matplotlib import fig_to_html
from vcm.calc.metrics import *


# https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7
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


def plot_vertical_accuracy_levels(truth, pred):
    thresholds = np.logspace(-50, -1, 100)
    for k, thresh in enumerate(thresholds):
        for func in [
            accuracy,
            precision,
            recall,
            true_positive_rate,
            false_positive_rate,
            f1_score,
        ]:
            yield func.__name__, (thresh,), func(
                truth != 0, numpy.abs(pred) > thresh, global_mean
            )


def global_mean(x):
    return x.mean(["time", "tile", "x", "y"])


def plot_profile_z_coord(x, label=""):
    x.drop("z").plot(y="z", yincrease=False, label=label)


def plot_2d_z(x, **kwargs):
    return x.drop("z").plot(yincrease=False, y="z", **kwargs)


def interpolate_onto_var(ds, variable, values, dim):
    return vcm.interpolate_1d(
        xr.DataArray(values, dims=[variable]), ds[variable], ds.drop(variable), dim=dim
    )


# %%
url = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-03/limit-tests-limiter-all-loss-rnn-7ef273-10d-88ef76-offline"
ds = open_rundir(url)
ds = ds.isel(time=slice(0, 8))

# %%

#%%

# %%

sections = {}


def classification_diags(ds, tendency, field):

    section_header = f"Metrics for {tendency.__name__}, {field}"

    figures = sections.setdefault(section_header, [])

    def insert_fig(key):
        plt.title(key)
        figures.append(fig_to_html(plt.gcf()))
        plt.close(plt.gcf())

    truth = tendency(ds, field, "physics")
    pred = tendency(ds, field, "emulator")

    combined = vcm.combine_array_sequence(
        plot_vertical_accuracy_levels(truth, pred), labels=["threshold"]
    ).load()

    # %%

    plot_profile_z_coord(global_mean(truth))
    plot_profile_z_coord(global_mean(pred))
    plt.legend()
    insert_fig("Mean")

    # %%
    fraction_non_zero = global_mean(truth != 0)
    plot_profile_z_coord(fraction_non_zero.rename("fraction non zero"))
    insert_fig("Fraction of truth non-zero")

    # %%
    i = interpolate_onto_var(combined, "recall", np.linspace(0, 1, 40), "threshold")
    plot_2d_z(i.precision, cmap=plt.get_cmap("viridis", 10))
    insert_fig("precision-recall")

    # %%
    i = interpolate_onto_var(
        combined, "false_positive_rate", np.linspace(0, 1, 40), "threshold"
    )
    plot_2d_z(i.true_positive_rate, vmin=0.4, vmax=1, cmap=plt.get_cmap("Blues", 6))
    insert_fig("ROC")

    # %%
    plot_profile_z_coord(
        i.true_positive_rate.fillna(0).mean("false_positive_rate").rename("AUC")
    )
    plt.axvline(0.5, linestyle="--")
    plt.xticks(np.r_[0:1:0.1])
    plt.grid()
    insert_fig("AUC")


FIELDS = ["cloud_water", "air_temperature", "specific_humidity"]
TENDENCIES = [
    tendencies.total_tendency,
    tendencies.gscond_tendency,
    tendencies.precpd_tendency,
]

for field in FIELDS:
    for tendency in TENDENCIES:
        classification_diags(ds, tendency, field)


import sys

html = report.create_html(
    sections,
    title="Classification metrics",
    metadata={"rundir": url, "times": len(ds.time), "argv": sys.argv},
)

with open("index.html", "w") as f:
    f.write(html)
# %%
