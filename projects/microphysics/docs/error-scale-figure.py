import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import vcm
from black import os
from fv3fit.train_microphysics import TrainConfig


def mean_square(x):
    return np.mean(x ** 2)


def open_model(path):
    fs = vcm.get_fs(path)
    with tempfile.TemporaryDirectory() as dir_:
        lpath = dir_ + "/model.tf"
        fs.get(path, lpath, recursive=True)
        return tf.keras.models.load_model(lpath)


def get_errors(model, train_set) -> pd.DataFrame:
    predictions = model(train_set)

    analysis = train_set.copy()
    for key in predictions:
        analysis["predicted_" + key] = predictions[key]

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

    return df


def plot(df, title):

    t_bins = np.arange(170, 320, 2.5)
    membership = pd.cut(
        df["air_temperature_input"], t_bins, labels=(t_bins[1:] + t_bins[:-1]) / 2
    )

    mean_square_df = df.groupby(membership).aggregate(mean_square)

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
        ax.legend(loc="lower right")

    plot_field("cloud_water_mixing_ratio", axs[1], "(kg/kg)^2")
    axs[1].set_ylim(bottom=1e-15)

    plot_field("air_temperature", axs[2], "K^2")
    plot_field("specific_humidity", axs[3], "(kg/kg)^2")
    axs[-1].set_xlabel("Temperature (K)")
    fig.suptitle(title)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # ERROR SCALE
    ERROR_SCALE_OUTPUT_DIR = "figs/error-scale"
    ERROR_SCALE_MODELS = [
        # "gs://vcm-ml-experiments/microphysics-emulation/2022-01-18/rnn-predict-gscond-3f77ec",
        # "gs://vcm-ml-experiments/microphysics-emulation/2022-01-27/rnn-gscond-cloudtdep-cbfc4a",
        # "gs://vcm-ml-experiments/microphysics-emulation/2022-01-27/rnn-gscond-alltdep-c9af46",
        "gs://vcm-ml-experiments/microphysics-emulation/2022-02-01/rnn-alltdep-47ad5b-de512-lr0.0002-login",  # noqa
    ]
    for model_url in ERROR_SCALE_MODELS:
        if "checkpoints" in model_url:
            # model_root = re.sub(r"/checkpoints/.*$", "", model_url)
            pass
        else:
            model_root = model_url
            model_url = model_url + "/model.tf"

        config = TrainConfig.from_yaml_path(model_root + "/config.yaml")
        train_ds = config.open_dataset(config.train_url, 10, config.model_variables)
        train_set = next(iter(train_ds.batch(180000)))
        model = tf.keras.models.load_model(model_url)
        df = get_errors(model, train_set)
        model_id = os.path.basename(model_root)
        plot(df, model_id)
        out = os.path.join(ERROR_SCALE_OUTPUT_DIR, model_id + ".png")
        plt.savefig(out)
