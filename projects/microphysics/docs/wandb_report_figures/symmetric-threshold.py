"""
# Will thresholding small tendencies reduce the cloud bias?

"""
from fv3net.diagnostics.prognostic_run.emulation import tendencies
from fv3net.diagnostics.prognostic_run.emulation.single_run import open_rundir

import vcm
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import report

from joblib import Memory

# cache to use for quick regeneration of figs
cache = Memory(location="~/.cache-joblib")
xr.set_options(keep_attrs=True)
plt.style.use(["seaborn-talk", "seaborn-whitegrid", "tableau-colorblind10"])

# parameters
URL = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-03/limit-tests-limiter-all-loss-rnn-7ef273-10d-88ef76-offline"  # noqa
field = "cloud_water"


def matplotlib_png(fig, **kwargs):
    ret = report.MatplotlibFigure(fig)
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

    ds = vcm.combine_array_sequence(_gen(), labels=["threshold"],).load()
    ds.threshold.attrs["units"] = truth.units
    return ds


def smooth(x, n=1):
    orig = x
    for i in range(n):
        x = np.convolve(np.pad(x, 1, mode="edge"), [1 / 4, 2 / 4, 1 / 4], mode="valid")
    return xr.DataArray(
        x, dims=orig.dims, coords=orig.coords, name=orig.name, attrs=orig.attrs
    )


def compute_optimal_threshold(data):
    """average the bias and rmse skill scores and smooth a bit"""

    def mean(a, b):
        return (a + b) / 2

    combined = mean(data.skill, data.bias_score)
    optimal_threshold = combined.threshold[combined.argmax("threshold")]
    return smooth(optimal_threshold, n=4)


@cache.cache
def get_data(url):
    ds = open_rundir(url)
    return ds.isel(time=slice(0, 8))


@cache.cache
def cache_stats(url, field, tendency):
    """v3"""
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
    data.bias.drop("z").plot(y="z", xscale="log", yincrease=False)
    optimal_threshold.drop("z").plot(y="z")

    fig2 = plt.figure()
    data.skill.drop("z").plot(y="z", xscale="log", vmax=1, vmin=-1, yincrease=False)
    optimal_threshold.drop("z").plot(y="z")

    fig3 = plt.figure(figsize=(6 / 1.61, 6))
    bias_optimal = smooth(data.bias.interp(threshold=optimal_threshold), n=2)
    bias_original = smooth(data.bias.isel(threshold=0), n=3)
    bias_optimal.drop("z").plot(y="z", yincrease=False, label="optimal threshold")
    bias_original.drop("z").plot(y="z", yincrease=False, label="no threshold")
    plt.legend()
    plt.title(f"Bias (smoothed)")
    plt.xlabel(f"{bias_optimal.units}")

    plt.close("all")
    return (
        f"{field} {tendency.__name__}",
        [matplotlib_png(fig), matplotlib_png(fig2), matplotlib_png(fig3)],
    )


def field_plots(url, field):
    yield report_(url, field, tendencies.gscond_tendency)
    yield report_(url, field, tendencies.precpd_tendency)


def all_plots(url):
    yield "Discussion", [report.RawHTML(__doc__)]
    for field in ["cloud_water", "air_temperature", "specific_humidity"]:
        yield from field_plots(url, field)


# %%
html = report.create_html(
    title="Symmetric Thresholding", sections=dict(all_plots(URL)), metadata={"url": URL}
)
# import pathlib
# pathlib.Path("local.html").write_text(html)
report.upload(html)
