import sys

import matplotlib.pyplot as plt
import numpy
import numpy as np
import report
import vcm
import xarray as xr
import xhistogram.xarray
import fv3viz
from vcm import accuracy, f1_score, false_positive_rate, precision, true_positive_rate

from fv3net.diagnostics.prognostic_run.emulation import tendencies
from fv3net.diagnostics.prognostic_run.emulation.single_run import open_rundir
from fv3net.diagnostics.prognostic_run.views.matplotlib import fig_to_html

fv3viz.use_colorblind_friendly_style()


def temperature_average(temperature, arr):
    bins = np.linspace(150, 300, 50)
    dim = set(arr.dims) & {"time", "tile", "y", "x"}
    count = xhistogram.xarray.histogram(
        temperature, bins=[bins], weights=xr.ones_like(arr), dim=dim
    )
    return (
        xhistogram.xarray.histogram(temperature, bins=[bins], weights=arr, dim=dim)
        / count
    )


def plot_vertical_accuracy_levels(truth, pred, avg):
    thresholds = np.logspace(-50, -1, 20)
    for thresh in thresholds:
        for func in [
            accuracy,
            precision,
            true_positive_rate,
            false_positive_rate,
            f1_score,
        ]:
            yield func.__name__, (thresh,), func(
                truth != 0, numpy.abs(pred) > thresh, avg
            )


def global_mean(x):
    return x.mean(["time", "tile", "x", "y"])


def plot_profile_z_coord(x, label=""):
    return x.plot(label=label)
    x.drop("z").plot(y="z", yincrease=False, label=label)


def plot_2d_z(x, **kwargs):
    return x.plot(**kwargs)
    return x.drop("z").plot(yincrease=False, x="z", **kwargs)


def interpolate_onto_var(ds, variable, values, dim):
    return vcm.interpolate_1d(
        xr.DataArray(values, dims=[variable]), ds[variable], ds.drop(variable), dim=dim
    )


# %%

sections = {}


def plot_global_means(truth, pred, avg):
    plot_profile_z_coord(avg(truth))
    plot_profile_z_coord(avg(pred))
    plt.legend()


def plot_fraction_nonzero(truth, avg):
    fraction_non_zero = avg(truth != 0)
    plot_profile_z_coord(fraction_non_zero.rename("fraction non zero"))


def temperature_z_binned_diags(
    ds, field="cloud_water", tendency=tendencies.precpd_tendency
):
    figures = []

    def insert_fig(key):
        plt.title(key)
        figures.append(fig_to_html(plt.gcf()))
        plt.close(plt.gcf())

    truth = tendency(ds, field, "physics")
    pred = tendency(ds, field, "emulator")
    temperature = ds.air_temperature

    def avg(x):
        return temperature_average(temperature, x)

    bins = np.linspace(150, 300, 50)
    count = xhistogram.xarray.histogram(
        ds.air_temperature, bins=[bins], dim=["time", "tile", "x", "y"]
    )
    count.drop("z").plot()
    insert_fig("histogram")

    # plot_global_means(truth, pred, avg)
    avg(truth).drop("z").plot()
    insert_fig("Mean")

    avg(pred).drop("z").plot()
    insert_fig("mean pred")

    (avg(pred) - avg(truth)).drop("z").plot(vmax=1e-10)
    insert_fig("bias")

    def plot_fraction_nonzero(truth, avg):
        avg(truth).plot(vmin=0, vmax=1)

    plot_fraction_nonzero(truth != 0, avg)
    insert_fig("Fraction of truth non-zero")

    plot_fraction_nonzero(truth > 0, avg)
    insert_fig("Fraction of true  > 0")

    plot_fraction_nonzero(pred > 0, avg)
    insert_fig("Fraction of pred  > 0")

    global_mean((pred - truth).where(pred > 0, 0)).plot(
        label="bias from positive points"
    )
    global_mean((pred - truth).where(pred <= 0, 0)).plot(
        label="bias from non-positive points"
    )
    global_mean(pred - truth).plot(label="bias")
    plt.legend()
    insert_fig("Bias from pred > 0")

    return figures


def classification_diags(ds, tendency, field):
    figures = []

    def insert_fig(key):
        plt.title(key)
        figures.append(fig_to_html(plt.gcf()))
        plt.close(plt.gcf())

    truth = tendency(ds, field, "physics")
    pred = tendency(ds, field, "emulator")

    combined = vcm.combine_array_sequence(
        plot_vertical_accuracy_levels(truth, pred, global_mean), labels=["threshold"]
    ).load()

    def plot_precision(combined):
        i = interpolate_onto_var(
            combined, "true_positive_rate", np.linspace(0, 1, 40), "threshold"
        )
        plot_2d_z(i.precision, cmap=plt.get_cmap("viridis", 10))
        insert_fig("precision-recall")

    def plot_roc(combined):
        i = interpolate_onto_var(
            combined, "false_positive_rate", np.linspace(0, 1, 40), "threshold"
        )
        plot_2d_z(i.true_positive_rate, vmin=0.4, vmax=1, cmap=plt.get_cmap("Blues", 6))
        insert_fig("ROC")

    def plot_auc(combined):
        i = interpolate_onto_var(
            combined, "false_positive_rate", np.linspace(0, 1, 40), "threshold"
        )
        i.true_positive_rate.fillna(0).mean("false_positive_rate").rename("AUC").plot()
        plt.axhline(0.5, linestyle="--")
        plt.yticks(np.r_[0:1:0.1])
        plt.grid()
        insert_fig("AUC")

    plot_precision(combined)
    plot_roc(combined)
    plot_auc(combined)
    return figures


FIELDS = ["cloud_water", "air_temperature", "specific_humidity"]
TENDENCIES = [
    tendencies.total_tendency,
    tendencies.gscond_tendency,
    tendencies.precpd_tendency,
]

FIELDS = ["cloud_water"]
TENDENCIES = [tendencies.precpd_tendency]

url = sys.argv[1]
ds = open_rundir(url)
ds = ds.isel(time=slice(0, 8))


sections["Cloud Precipitation Tendency Diagnostics"] = temperature_z_binned_diags(ds)

for field in FIELDS:
    for tendency in TENDENCIES:
        sections[f"{field}, {tendency.__name__}"] = classification_diags(
            ds, tendency, field
        )

html = report.create_html(
    sections,
    title="Classification metrics",
    metadata={"rundir": url, "times": len(ds.time), "argv": sys.argv},
)

with open("index.html", "w") as f:
    f.write(html)
