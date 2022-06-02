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
        "cloud_water_mixing_ratio_after_gscond",
        # "air_temperature_after_gscond",
        # "air_temperature_after_precpd",
        # "specific_humidity_after_precpd",
        # "cloud_water_mixing_ratio_after_precpd",
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


def group(gen):
    return [report.MatplotlibFigure(fig) for _, fig in gen]


def open_truth_prediction(
    test_url="/Users/noahb/data/vcm-ml-experiments/microphysics-emulation/2021-11-24/microphysics-training-data-v3-training_netcdfs/test",
    model_path="/Users/noahb/workspace/ai2cm/fv3net/model.tf",
    n_samples=200_000,
    rectified=True,
):

    model = tf.keras.models.load_model(model_path)
    tfds = open_data(test_url)
    truth_pred = tfds.map(lambda x: (x, model(x)))
    truth_dict, pred_dict = next(iter(truth_pred.unbatch().batch(n_samples)))
    truth = tensordict_to_dataset(truth_dict)
    pred = tensordict_to_dataset(pred_dict)
    pred["cloud_water_mixing_ratio_after_gscond"] = truth[
        "cloud_water_mixing_ratio_input"
    ] - (pred["specific_humidity_after_gscond"] - truth["specific_humidity_input"])
    # rectify
    if rectified:
        qc_name = "cloud_water_mixing_ratio_after_gscond"
        qc = pred[qc_name]
        pred[qc_name] = qc.where(qc > 0, 0)
    return truth, pred


def plots(truth, pred):
    yield class_fractions(truth)
    yield "bias", group(bias_plots(truth, pred))


def class_fractions(truth):
    classes = classify(
        truth.cloud_water_mixing_ratio_input,
        truth.cloud_water_mixing_ratio_after_gscond,
    )

    return plot_class_fractions_by_z(classes)


def plot_class_fractions_by_z(classes):
    classes.mean("sample").to_dataframe().plot.area().legend(loc="upper left")
    plt.ylabel("Fraction")
    plt.xlabel("vertical index (0=TOA)")
    plt.grid()
    return "fraction", [report.MatplotlibFigure(plt.gcf())]


def bias_plots(truth, pred):

    classes = classify(
        truth.cloud_water_mixing_ratio_input,
        truth.cloud_water_mixing_ratio_after_gscond,
    )
    assert set(classes.to_array().sum("variable").values.ravel()) == {1}
    plot_metrics_on_classes(classes, truth, pred)


def plot_metrics_on_classes(classes, truth, pred):
    plt.figure()
    error = pred - truth
    for v in classes:
        (error.cloud_water_mixing_ratio_after_gscond / 900).where(classes[v], 0).mean(
            "sample"
        ).plot(label=v)
    (error.cloud_water_mixing_ratio_after_gscond / 900).mean("sample").plot(
        label="net", color="black", linestyle=":"
    )
    plt.xlabel("vertical index (0=TOA)")
    plt.ylabel("Total ZC cloud tendency (kg/kg/s)")
    plt.title("Bias | category * p(category)")
    plt.grid()
    plt.legend()
    yield "bias", plt.gcf()

    bias = metric_on_classes(
        (
            truth.cloud_water_mixing_ratio_after_gscond
            - truth.cloud_water_mixing_ratio_input
        )
        / 900,
        (
            pred.cloud_water_mixing_ratio_after_gscond
            - truth.cloud_water_mixing_ratio_input
        )
        / 900,
        classes,
        lambda x, y, mean: mean(y - x),
    )
    bias.to_dataframe().plot.line()
    plt.xlabel("vertical index (0=TOA)")
    plt.ylabel("Bias (kg/kg/s)")
    plt.title("Bias | category")
    plt.grid()
    yield "bias", plt.gcf()

    mses = np.sqrt(
        metric_on_classes(
            (
                truth.cloud_water_mixing_ratio_after_gscond
                - truth.cloud_water_mixing_ratio_input
            )
            / 900,
            (
                pred.cloud_water_mixing_ratio_after_gscond
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

    def skill_score(truth, pred, mean):
        sse = mean((truth - pred) ** 2)
        ss = mean(pred ** 2)
        return 1 - sse / ss

    mses = np.sqrt(
        metric_on_classes(
            (
                truth.cloud_water_mixing_ratio_after_gscond
                - truth.cloud_water_mixing_ratio_input
            )
            / 900,
            (
                pred.cloud_water_mixing_ratio_after_gscond
                - truth.cloud_water_mixing_ratio_input
            )
            / 900,
            classes,
            skill_score,
        )
    )
    mses.to_dataframe().plot.line()
    plt.ylim([0, 1])
    plt.xlabel("vertical index (0=TOA)")
    plt.ylabel("R2")
    plt.title("R2 | category")
    plt.grid()
    yield "r2", plt.gcf()


if __name__ == "__main__":
    plt.style.use(["tableau-colorblind10", "seaborn-talk"])
    test_url = "gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/microphysics-training-data-v4/test"
    model_path = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-13/gscond-only-dense-local-nfiles1980-41b1c1-v1/model.tf"
    truth, pred = open_truth_prediction(test_url, model_path)
    html = report.create_html(
        dict(plots(truth, pred)),
        title="Category analysis",
        metadata={"model_path": model_path, "test_url": test_url, "script": __file__},
    )
    report.upload(html)
    import pathlib

    pathlib.Path("report.html").write_text(html)
