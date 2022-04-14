# flake8: noqa
from audioop import bias
import os
import xarray as xr
import numpy as np
import vcm
from fv3fit.emulation.data.load import nc_dir_to_tfdataset
from fv3fit.train_microphysics import TransformConfig
import tensorflow as tf
import matplotlib.pyplot as plt
import report


def tensordict_to_dataset(x):
    """convert a tensor dict into a xarray dataset and flip the vertical coordinate"""

    def _get_dims(val):
        n, feat = val.shape
        if feat == 1:
            return (["sample"], val[:, 0].numpy())
        else:
            return (["sample", "z"], val[:, ::-1].numpy())

    return xr.Dataset({key: _get_dims(val) for key, val in x.items()})


def classify(cloud_in, cloud_out):
    """

    digraph G {
        {node[shape=point] root cloud_2 somecloud nocloud}
        splines=false;


        root -> nocloud [label="in = 0"];
        root -> somecloud [label="in != 0"];


        somecloud -> destroyed [label="out = 0"]
        somecloud -> cloud_2 [label="out != 0"]

        nocloud -> nothing [label="out = 0"]
        nocloud -> "cloud created" [label="out != 0"]

        destroyed -> "small cloud \ndestroyed" [label="small change"]
        destroyed -> "large cloud\ndestroyed" [label="else"]

        cloud_2 -> "slightly changed" [label="small change"]
        cloud_2 -> "no change" [label="else"]
    }

    """
    small_change = np.abs(cloud_in - cloud_out) < 1e-9
    out_0 = np.abs(cloud_out) < 1e-12
    in_0 = np.abs(cloud_in) < 1e-12

    return xr.Dataset(
        {
            "slight change": (~in_0) & (~out_0) & (small_change),
            "large change": (~in_0) & (~out_0) & (~small_change),
            "small cloud destroyed": (~in_0) & (out_0) & (small_change),
            "large cloud destroyed": (~in_0) & (out_0) & (~small_change),
            "cloud created": in_0 & (~out_0),
            "nothing": in_0 & out_0,
        }
    )


def open_data(url: str) -> tf.data.Dataset:
    variables = [
        "latitude",
        "longitude",
        "pressure_thickness_of_atmospheric_layer",
        "air_pressure",
        "surface_air_pressure",
        "air_temperature_input",
        "specific_humidity_input",
        "cloud_water_mixing_ratio_input",
        "air_temperature_after_last_gscond",
        "specific_humidity_after_last_gscond",
        "surface_air_pressure_after_last_gscond",
        "specific_humidity_after_gscond",
        "air_temperature_after_gscond",
        "air_temperature_after_precpd",
        "specific_humidity_after_precpd",
        "cloud_water_mixing_ratio_after_precpd",
        "total_precipitation",
        "ratio_of_snowfall_to_rainfall",
        "tendency_of_rain_water_mixing_ratio_due_to_microphysics",
        "time",
    ]
    data_transform = TransformConfig(
        antarctic_only=False,
        use_tensors=True,
        vertical_subselections=None,
        derived_microphys_timestep=900,
    )
    # TODO change default of get_pipeline to get all the variables
    # will allow deleting the code above
    return nc_dir_to_tfdataset(url, data_transform.get_pipeline(variables))


def metric_on_classes(truth, pred, classes, metric):
    out = {}
    for v in classes:

        def avg(x):
            return vcm.weighted_average(x, classes[v], dims=["sample"])

        out[v] = metric(truth, pred, avg)
    return xr.Dataset(out)


def threshold_analysis(truth, pred):

    c_in = truth.cloud_water_mixing_ratio_input
    c_0 = np.abs(c_in) >= 1e-12
    c_1 = np.abs(truth.cloud_water_mixing_ratio_after_precpd) >= 1e-12
    destroyed = c_0 & (~c_1)

    fractional_change = (pred.cloud_water_mixing_ratio_after_precpd - c_in) / c_in

    def avg(x):
        return vcm.weighted_average(x, c_0, dims=["sample", "z"])

    def _gen():
        thresholds = np.linspace(-0.5, 1.5, 100)
        for thresh in thresholds:
            for func in [vcm.false_positive_rate, vcm.true_positive_rate, vcm.accuracy]:
                yield func.__name__, (thresh,), func(
                    destroyed, fractional_change < -thresh, avg
                )

    return vcm.combine_array_sequence(_gen(), labels=["threshold"]).assign(
        prob=avg(destroyed)
    )


def plot_threshold_analysis(truth, pred):
    acc = threshold_analysis(truth, pred)
    acc = acc.swap_dims({"threshold": "false_positive_rate"}).sortby(
        "false_positive_rate"
    )
    plt.figure()
    acc.true_positive_rate.plot()
    auc = acc.true_positive_rate.integrate("false_positive_rate")
    plt.title(f"ROC AUC={float(auc):3f} Prob={float(acc.prob):.3f}")
    yield "roc", plt.gcf()
    plt.figure()
    acc.threshold.plot()
    yield "threshold", plt.gcf()


def group(gen):
    return [report.MatplotlibFigure(fig) for _, fig in gen]


def main(
    test_url="/Users/noahb/data/vcm-ml-experiments/microphysics-emulation/2021-11-24/microphysics-training-data-v3-training_netcdfs/test",
    model_path="/Users/noahb/workspace/ai2cm/fv3net/model.tf",
    n_samples=20_000,
):

    model = tf.keras.models.load_model(model_path)
    tfds = open_data(test_url)
    truth_pred = tfds.map(lambda x: (x, model(x)))
    truth_dict, pred_dict = next(iter(truth_pred.unbatch().batch(n_samples)))
    truth = tensordict_to_dataset(truth_dict)
    pred = tensordict_to_dataset(pred_dict)

    yield class_fractions(truth)
    yield "threshold analysis", group(plot_threshold_analysis(truth, pred))
    yield "bias", group(bias_plots(truth, pred))
    yield "bias theshold 20% decrease", group(bias_plots_thresholded(truth, pred, -0.2))
    yield "bias theshold 50% decrease", group(bias_plots_thresholded(truth, pred, -0.5))


def class_fractions(truth):
    classes = classify(
        truth.cloud_water_mixing_ratio_input,
        truth.cloud_water_mixing_ratio_after_precpd,
    )
    classes.mean("sample").to_dataframe().plot.area().legend(loc="upper left")
    plt.ylabel("Fraction")
    plt.xlabel("vertical index (0=TOA)")
    plt.grid()
    return "fraction", [report.MatplotlibFigure(plt.gcf())]


def bias_plots_thresholded(truth, pred, threshold):
    c_in = truth.cloud_water_mixing_ratio_input
    c_0 = np.abs(c_in) >= 1e-12
    fractional_change = (pred.cloud_water_mixing_ratio_after_precpd - c_in) / c_in
    new_precpd = xr.where(
        c_0 & (fractional_change < threshold),
        0,
        pred.cloud_water_mixing_ratio_after_precpd,
    )
    pred_thresholded = pred.assign(cloud_water_mixing_ratio_after_precpd=new_precpd)
    yield from bias_plots(truth, pred_thresholded)


def bias_plots(truth, pred):

    error = pred - truth
    classes = classify(
        truth.cloud_water_mixing_ratio_input,
        truth.cloud_water_mixing_ratio_after_precpd,
    )
    assert set(classes.to_array().sum("variable").values.ravel()) == {1}

    plt.figure()
    for v in classes:
        (error.cloud_water_mixing_ratio_after_precpd / 900).where(classes[v], 0).mean(
            "sample"
        ).plot(label=v)
    (error.cloud_water_mixing_ratio_after_precpd / 900).mean("sample").plot(
        label="net", color="black", linestyle=":"
    )
    plt.xlabel("vertical index (0=TOA)")
    plt.ylabel("Total ZC cloud tendency (kg/kg/s)")
    plt.title("Bias | category * p(category)")
    plt.grid()
    plt.legend()
    yield "bias", plt.gcf()

    mses = np.sqrt(
        metric_on_classes(
            (
                truth.cloud_water_mixing_ratio_after_precpd
                - truth.cloud_water_mixing_ratio_input
            )
            / 900,
            (
                pred.cloud_water_mixing_ratio_after_precpd
                - truth.cloud_water_mixing_ratio_input
            )
            / 900,
            classes,
            vcm.mean_squared_error,
        )
    )
    mses.to_dataframe().plot.line()
    plt.xlabel("vertical index (0=TOA)")
    plt.ylabel("RMSE (kg/kg/s)")
    plt.title("RMSE | category")
    plt.grid()
    yield "mse", plt.gcf()


plt.style.use(["tableau-colorblind10", "seaborn-talk"])
test_url = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-17/online-12hr-cycle-v3-online/artifacts/20160611.000000/netcdf_output"
model_path = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-02/limit-tests-limiter-all-loss-rnn-7ef273/model.tf"
# model_path = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-02/limit-tests-all-loss-rnn-7ef273/model.tf"
html = report.create_html(
    dict(main(model_path=model_path, test_url=test_url)),
    title="Category analysis",
    metadata={"model_path": model_path, "test_url": test_url, "script": __file__},
)
report.upload(html)
# import pathlib
# pathlib.Path("report.html").write_text(html)
