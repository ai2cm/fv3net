"""
# Will thresholding small tendencies reduce the cloud bias?
{{date}}
- Author: Noah Brenowitz

{{url}}

"""
from datetime import datetime, date
from fv3net.diagnostics.prognostic_run.emulation import tendencies
from fv3net.diagnostics.prognostic_run.emulation.single_run import open_rundir

import vcm
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import datapane
import report

from joblib import Memory

# cache to use for quick regeneration of figs
cache = Memory(location="~/.cache-joblib")

# parameters
URL = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-03/limit-tests-limiter-all-loss-rnn-7ef273-10d-88ef76-offline"  # noqa
field = "cloud_water"


def matplotlib_png(fig, **kwargs):
    ret = datapane.HTML(report.MatplotlibFigure(fig, dpi=75, format="jpg"))
    plt.close(fig)
    return ret


def global_mean(x):
    return x.mean(["time", "tile", "x", "y"])


def bias_score(truth, pred, avg):
    num = np.abs(avg(pred - truth))
    denom = np.abs(avg(truth)) + np.abs(avg(pred))
    return 1 - xr.where(denom != 0, num / denom, 0)


def bias(truth, pred, avg):
    return avg(pred - truth)


def rmse(truth, pred, avg):
    return np.sqrt(avg((truth - pred) ** 2))


def mse(truth, pred, avg):
    return avg((truth - pred) ** 2)


def skill(truth, pred, avg):
    denom = mse(truth, 0, avg) / 2 + mse(0, pred, avg) / 2
    num = mse(truth, pred, avg)
    return 1 - xr.where(denom != 0, num / denom, 0)


def statistics_across_thresholds(
    truth, pred, average=global_mean, statistics=[bias, bias_score, rmse, skill],
):
    def _gen():
        thresholds = np.logspace(-12, -5, 50)
        for thresh in thresholds:
            for func in statistics:
                pred_thresholded = pred.where(np.abs(pred) > thresh, 0)
                yield func.__name__, (thresh,), func(truth, pred_thresholded, average)

    return vcm.combine_array_sequence(_gen(), labels=["threshold"],).load()


def compute_optimal_threshold(data):
    """average the bias and rmse skill scores and smooth a bit"""

    def mean(a, b):
        return (a + b) / 2

    def smooth(x, n=1):
        orig = x
        for i in range(n):
            x = np.convolve(
                np.pad(x, 1, mode="edge"), [1 / 4, 2 / 4, 1 / 4], mode="valid"
            )
        return xr.DataArray(x, dims=orig.dims, coords=orig.coords, name=orig.name)

    combined = mean(data.skill, data.bias_score)
    optimal_threshold = combined.threshold[combined.argmax("threshold")]
    return smooth(optimal_threshold, n=4)


@cache.cache
def get_data(url):
    ds = open_rundir(url)
    return ds.isel(time=slice(0, 8))


@cache.cache
def cache_stats(url, field, tendency):
    ds = get_data(url)
    return statistics_across_thresholds(
        tendency(ds, field, "physics"),
        tendency(ds, field, "emulator"),
        average=global_mean,
    )


def report_(url, field, tendency):
    data = cache_stats(url, field, tendency)
    optimal_threshold = compute_optimal_threshold(data)

    fig = plt.figure()
    data.bias.drop("z").plot(y="z", xscale="log")
    optimal_threshold.drop("z").plot(y="z")

    fig2 = plt.figure()
    data.skill.drop("z").plot(y="z", xscale="log", vmax=1, vmin=-1)
    optimal_threshold.drop("z").plot(y="z")
    plt.close("all")
    return datapane.Group(
        datapane.Text(f"## {tendency.__name__}"),
        datapane.Group(matplotlib_png(fig), matplotlib_png(fig2), columns=2,),
    )


def field_plots(url, field):
    return datapane.Group(
        report_(url, field, tendencies.gscond_tendency),
        report_(url, field, tendencies.precpd_tendency),
        label=field,
    )


def all_plots(url):
    return datapane.Select(
        field_plots(url, "cloud_water"),
        field_plots(url, "air_temperature"),
        field_plots(url, "specific_humidity"),
        type=datapane.SelectType.TABS,
    )


# %%
date = date.today().isoformat()
datapane.Report(
    datapane.Text(__doc__).format(date=datetime.now().isoformat(), url=URL),
    all_plots(URL),
).save("local.thml")
