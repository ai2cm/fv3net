#!/usr/bin/env python

# Make predictions of the coarsened fine clouds from the nudged coarse state,
# and write them to disk


import fv3fit
import intake
import cftime

MODEL_PATH = "gs://vcm-ml-experiments/cloud-ml/2022-12-01/fine-cloud-rf-incloud-local/trained_model"  # noqa: E501
NUDGED_COARSE_STATE_PATH = (
    "gs://vcm-ml-experiments/cloud-ml/2022-09-14/fine-coarse-3d-fields.zarr"
)
TIME_START = cftime.DatetimeJulian(2016, 8, 5, 1, 0, 0, 0)
TIME_END = cftime.DatetimeJulian(2016, 8, 6, 1, 0, 0, 0)
TIME_SEGMENT_LENGTH = 1
OUTPUT_PATH = (
    "gs://vcm-ml-experiments/cloud-ml/2022-12-01/predicted-fine-cloud-fields-v2.zarr"
)


def get_time_slices(n_times, segment_length):
    time_slices = []
    for i_start in range(0, n_times, segment_length):
        time_slices.append(slice(i_start, i_start + segment_length))
    return time_slices


if __name__ == "__main__":
    model = fv3fit.load(MODEL_PATH)
    coarse_ds = intake.open_zarr(NUDGED_COARSE_STATE_PATH, consolidated=True).to_dask()
    inputs = coarse_ds[model.input_variables].sel(time=slice(TIME_START, TIME_END))
    # this is a hack, because otherwise fv3fit will not allow grid-cell predictions
    inputs = inputs.rename({"z": "hidden_vertical_dim"})
    time_slices = get_time_slices(len(inputs.time), TIME_SEGMENT_LENGTH)
    input_0 = inputs.isel(time=time_slices[0])
    prediction_0 = model.predict(input_0)
    prediction_0.rename({"hidden_vertical_dim": "z"}).to_zarr(OUTPUT_PATH)
    for time_slice in time_slices[1:]:
        input_ = inputs.isel(time=time_slice)
        print(input_.time[0].item())
        prediction = model.predict(input_)
        prediction.rename({"hidden_vertical_dim": "z"}).to_zarr(
            OUTPUT_PATH, mode="a", append_dim="time"
        )
