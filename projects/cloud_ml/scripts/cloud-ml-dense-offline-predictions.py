#!/usr/bin/env python

# Make predictions of the coarsened fine clouds from the nudged coarse state,
# and write them to disk


import fv3fit
import intake
import cftime
import sys
import os
import yaml
import fsspec
import tensorflow as tf

MODEL_PATH = "gs://vcm-ml-experiments/cloud-ml/2022-12-17/fine-cloud-rf-incloud-binary10-local/trained_model"  # noqa: E501
NUDGED_COARSE_STATE_PATH = (
    "gs://vcm-ml-experiments/cloud-ml/2022-09-14/fine-coarse-3d-fields.zarr"
)
TIME_START = cftime.DatetimeJulian(2016, 8, 5, 1, 0, 0, 0)
TIME_END = cftime.DatetimeJulian(2016, 8, 6, 1, 0, 0, 0)
TIME_STRIDE = 1
OUTPUT_PATH = (
    "gs://vcm-ml-experiments/cloud-ml/2022-12-21/predicted-fine-cloud-fields-v4.zarr"
)
INPUT_VARIABLES = [
    "pressure",
    "relative_humidity",
    "air_temperature",
    "cloud_ice_mixing_ratio_coarse",
]
OUTPUT_VARIABLES = [
    "cloud_ice_mixing_ratio",
    "cloud_water_mixing_ratio",
    "cloud_amount",
]


def open_checkpoint_model(
    path, input_variables=INPUT_VARIABLES, output_variables=OUTPUT_VARIABLES
):
    tf_model = tf.keras.models.load_model(path, custom_objects={"custom_loss": None})
    fv3fit_model = fv3fit.PureKerasModel(
        input_variables,
        output_variables,
        tf_model,
        unstacked_dims=["some_dim",],
        n_halo=0,
    )
    return fv3fit_model


def maybe_modify_model_config(config_path):
    with fsspec.open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if list(config["unstacked_dims"]) == ["z"]:
        config["unstacked_dims"] = ("some_dim",)
        with fsspec.open(config_path, "w") as f:
            yaml.dump(config, f)
    return config


def get_time_steps(n_times, time_stride=None):
    time_slices = []
    for i_start in range(0, n_times, time_stride):
        time_slices.append(slice(i_start, i_start + 1))
    return time_slices


def get_config(path):
    if path is not None:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        return {}


def main(config):
    checkpoint_model = config.get("checkpoint_model", False)
    model_path = config.get("model_path", MODEL_PATH)
    output_path = config.get("output_path", OUTPUT_PATH)
    if checkpoint_model:
        model = open_checkpoint_model(model_path)
    else:
        config_path = os.path.join(model_path, "config.yaml")
        model_config = maybe_modify_model_config(config_path)
        print(model_config)
        model = fv3fit.load(model_path)
    coarse_ds = intake.open_zarr(
        config.get("coarse_nudged_state_path", NUDGED_COARSE_STATE_PATH),
        consolidated=True,
    ).to_dask()
    start_time = (
        cftime.DatetimeJulian(*config["start_time"])
        if "start_time" in config
        else TIME_START
    )
    end_time = (
        cftime.DatetimeJulian(*config["end_time"]) if "end_time" in config else TIME_END
    )
    inputs = coarse_ds[model.input_variables].sel(time=slice(start_time, end_time))
    time_slices = get_time_steps(
        len(inputs.time), time_stride=config.get("time_stride", TIME_STRIDE)
    )
    input_0 = inputs.isel(time=time_slices[0])
    prediction_0 = model.predict(input_0)
    prediction_0.to_zarr(output_path)
    for time_slice in time_slices[1:]:
        input_ = inputs.isel(time=time_slice)
        print(input_.time[0].item())
        prediction = model.predict(input_)
        prediction.to_zarr(output_path, mode="a", append_dim="time")


if __name__ == "__main__":
    if len(sys.argv[1]) > 1:
        config_path = sys.argv[1]
    else:
        config_path = None
    config = get_config(config_path)
    main(config)
