#!/usr/bin/env python

import cftime
import sys
import yaml
import xarray as xr
import fv3fit
from vcm import DerivedMapping

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


def get_config(path):
    if path is not None:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        return {}


def main(config):
    model_path = config["model_path"]
    output_path = config["output_path"]
    model = fv3fit.load(model_path)
    coarse_ds = xr.open_zarr(config["input_data_path"])
    inputs = DerivedMapping(coarse_ds).dataset(model.input_variables)
    start_time = cftime.DatetimeJulian(*config["start_time"])
    end_time = cftime.DatetimeJulian(*config["end_time"])
    time_stride = config.get("time_stride", None)
    inputs = inputs.sel(time=slice(start_time, end_time, time_stride))
    input_0 = inputs.isel(time=[0])
    print(input_0.time.item())
    prediction_0 = model.predict(input_0)
    prediction_0.to_zarr(output_path)
    for i_time in range(1, len(inputs.time)):
        input_ = inputs.isel(time=[i_time])
        print(input_.time.item())
        prediction = model.predict(input_)
        prediction.to_zarr(output_path, mode="a", append_dim="time")


if __name__ == "__main__":
    if len(sys.argv[1]) > 1:
        config_path = sys.argv[1]
    else:
        config_path = None
    config = get_config(config_path)
    main(config)
