# flake8: noqa
#%%
import numpy as np
import xarray as xr
import vcm
import matplotlib.pyplot as plt
from bias_categories import (
    open_truth_prediction,
    plot_class_fractions_by_z,
    plot_metrics_on_classes,
)
import pathlib
import report
import joblib

cache = joblib.Memory("/tmp/joblib")
open_truth_prediction = cache.cache(open_truth_prediction)

# plt.style.use(["tableau-colorblind10", "seaborn-talk"])


test_url = "gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/microphysics-training-data-v4/test"
model_path = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-13/gscond-only-dense-local-nfiles1980-41b1c1-v1/model.tf"
truth, pred = open_truth_prediction(
    test_url, model_path, rectified=True, n_samples=200_000
)
cloud_in = truth.cloud_water_mixing_ratio_input
cloud_out = truth.cloud_water_mixing_ratio_after_gscond
cloud_out_pred = pred.cloud_water_mixing_ratio_after_gscond
timestep = 900

tend = (cloud_out - cloud_in) / timestep
tend_pred = (cloud_out_pred - cloud_in) / timestep

mask = np.abs(cloud_in) > 1e-6


def plot_net_condensation_rate(mask):
    def average(x):
        return vcm.weighted_average(x, mask, ["sample"])

    scale = 1e3 * 86400

    average(tend * scale).plot(label="truth")
    average(tend_pred * scale).plot(label="pred")
    plt.grid()
    plt.ylabel("g/kg/d")
    plt.legend()
    plt.title("net condensation rate")


def classify(cloud_in, cloud_out):
    tend = (cloud_out - cloud_in) / timestep
    thresh = 1e-10
    some_cloud_out = np.abs(cloud_out) > 1e-15
    negative_tend = tend < -thresh
    return xr.Dataset(
        {
            "positive_tendency": tend > thresh,
            "zero_tendency": np.abs(tend) <= thresh,
            "zero_cloud": negative_tend & ~some_cloud_out,
            "negative_tendency": negative_tend & some_cloud_out,
        }
    )


def plot_p_vs_lat_fractions(truth, classes):
    merged = xr.merge([truth, classes])
    out = {}

    def gen():
        for key in classes:
            z = merged[key].astype(float).rename(key)
            pressure_int = vcm.interpolate_to_pressure_levels(
                z, merged["pressure_thickness_of_atmospheric_layer"], dim="z"
            ).rename(key)
            avg = vcm.zonal_average_approximate(
                np.rad2deg(merged.latitude), pressure_int
            ).rename(key)
            yield "fraction", (key,), avg

    ds = vcm.combine_array_sequence(gen(), labels=["class"])
    ds.fraction.plot(
        y="pressure", yincrease=False, col="class", col_wrap=2, figsize=(12, 6)
    )
    return "fraction", [report.MatplotlibFigure(plt.gcf())]


def matplotlib_figs():
    classes = classify(cloud_in, cloud_out)
    yield plot_class_fractions_by_z(classes)
    yield plot_p_vs_lat_fractions(truth, classes)
    for name, fig in plot_metrics_on_classes(classes, truth, pred):
        yield name, [report.MatplotlibFigure(fig)]


plt.style.use(["tableau-colorblind10", "seaborn-talk"])
test_url = "gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/microphysics-training-data-v4/test"
model_path = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-13/gscond-only-dense-local-nfiles1980-41b1c1-v1/model.tf"

html = report.create_html(
    dict(matplotlib_figs()),
    title="Output category analysis",
    metadata={"model_path": model_path, "test_url": test_url, "script": __file__},
)

# %%
pathlib.Path("report.html").write_text(html)
report.upload(html)
