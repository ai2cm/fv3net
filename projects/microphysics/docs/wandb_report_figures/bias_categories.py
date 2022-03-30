# flake8: noqa
import xarray as xr
import numpy as np
from fv3fit.emulation.data.load import nc_dir_to_tfdataset
from fv3fit.train_microphysics import TransformConfig
import tensorflow as tf
import matplotlib.pyplot as plt


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

    error = pred - truth
    classes = classify(
        truth.cloud_water_mixing_ratio_input,
        truth.cloud_water_mixing_ratio_after_precpd,
    )
    assert set(classes.to_array().sum("variable").values.ravel()) == {1}

    for v in classes:
        (error.cloud_water_mixing_ratio_after_precpd / 900).where(classes[v], 0).mean(
            "sample"
        ).plot(label=v)
    (error.cloud_water_mixing_ratio_after_precpd / 900).mean("sample").plot(
        label="net", color="black", linestyle=":"
    )
    plt.xlabel("vertical index (0=TOA)")
    plt.ylabel("Total ZC cloud tendency (kg/kg/s)")
    plt.title("Emulator bias within categories")
    plt.grid()
    plt.legend()
    yield "bias", plt.gcf()

    classes.mean("sample").to_dataframe().plot.area().legend(loc="upper left")
    plt.ylabel("Fraction")
    plt.xlabel("vertical index (0=TOA)")
    plt.grid()
    yield "fraction", plt.gcf()


plt.style.use(["tableau-colorblind10", "seaborn-talk"])
for key, fig in main():
    fig.savefig(f"{key}.png")

# config_url = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-02/limit-tests-limiter-all-loss-rnn-7ef273"
# test_url = "gs://vcm-ml-experiments/microphysics-emulation/2022-03-17/online-12hr-cycle-v3-online/artifacts/20160611.000000/netcdf_output"
