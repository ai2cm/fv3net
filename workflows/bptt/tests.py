from preprocessing import open_nudge, TrainingArrays, SAMPLE_DIM_NAME
from train import prepare_keras_arrays, build_model, compile_model, get_dim_lengths, get_stepwise_model, get_packers_and_scalers
from evaluate import arrays_to_dataset
import xarray as xr
import numpy as np
import tensorflow as tf
import fv3fit


# def test_sequential_model_same_answer_as_series_model():
#     with open("arrays/arrays_00000", "rb") as f:
#         arrays = TrainingArrays.load(f)
#     # arrays = open_nudge.__iter__.__next__()
#     n_input, n_state, n_window = get_dim_lengths(arrays)

#     (
#         input_packer, input_scaler, prognostic_packer, prognostic_scaler,
#         tendency_scaler
#     ) = get_packers_and_scalers(arrays)

#     units = 32
#     n_hidden_layers = 3
#     tendency_ratio = 0.2  # scaling between values and their tendencies
#     kernel_regularizer = None
#     timestep_seconds = 3 * 60 * 60

#     nz = int(n_state / 2.0)

#     # this model does not normalize, it acts on normalized data
#     model = build_model(
#         n_input,
#         n_state,
#         n_window,
#         units,
#         n_hidden_layers,
#         tendency_ratio,
#         kernel_regularizer,
#         timestep_seconds,
#     )
#     compile_model(model, prognostic_scaler)

#     stepwise_model = get_stepwise_model(model, n_input, n_state)

#     recurrent = fv3fit.keras.RecurrentModel(
#         SAMPLE_DIM_NAME,
#         input_variables=list(arrays.input_names),
#         model=stepwise_model,
#         input_packer=input_packer,
#         prognostic_packer=prognostic_packer,
#         input_scaler=input_scaler,
#         prognostic_scaler=prognostic_scaler,
#         train_timestep_seconds=timestep_seconds,
#     )

#     ds = arrays_to_dataset(arrays)
#     norm_input, norm_given_tendency, norm_prognostic_reference = prepare_keras_arrays(
#         arrays, input_scaler, tendency_scaler, prognostic_scaler, timestep_seconds
#     )
#     # ensure there is initialization info for the first tendency
#     assert not np.all(norm_given_tendency[:, 0, :] == 0.)
#     zero_tendency = np.zeros_like(norm_given_tendency)
#     zero_tendency[:, 0, :] = norm_given_tendency[:, 0, :]  # still need to initialize
#     output = prognostic_scaler.denormalize(model.predict([norm_input, zero_tendency]))
#     # ensure initial output is only model initialization
#     np.testing.assert_almost_equal(prognostic_scaler.normalize(output[:, 0, :]), norm_prognostic_reference[:, 0, :], decimal=6)
#     np.testing.assert_almost_equal(prognostic_scaler.normalize(output[:, 0, :]), timestep_seconds * norm_given_tendency[:, 0, :], decimal=5)

#     # remember the multistep model contains an "initialization" step
#     # the output of the multistep model from the first timestep is
#     # the state value at index 1. The state value at index 0 is the
#     # model initial state. Hence we compare the multistep t1-t0
#     # against the stepwise model's initial prediction

#     state_in = ds.isel(time=0)
#     forcing, state = recurrent._get_inputs(state_in)
#     np.testing.assert_almost_equal(norm_input[:, 0, :], forcing, decimal=5)
#     # input is the output from the previous timestep
#     np.testing.assert_almost_equal(prognostic_scaler.normalize(output[:, 0, :]), state, decimal=5)
#     # columnwise_stepwise_tendency_ds = recurrent.predict_columnwise(state_ds, feature_dim="z")
#     stepwise_tendency_ds = recurrent.predict(state_in)

#     np.testing.assert_almost_equal(
#         stepwise_tendency_ds["dQ1"].values * timestep_seconds,
#         output[:, 1, :nz] - output[:, 0, :nz],
#         decimal=5
#     )


# def test_arrays_integrate():
#     timestep_seconds = 3 * 60 * 60
#     with open("arrays/arrays_00000", "rb") as f:
#         arrays = TrainingArrays.load(f)

#     ds = arrays_to_dataset(arrays)

#     state_ds = ds.isel(time=0)
#     n_timesteps = len(ds["time"])
#     for i in range(n_timesteps):
#         print(f"Step {i+1} of {n_timesteps}")
#         forcing_ds = ds.isel(time=i)
#         # state at start of timestep should match forcing dataset state
#         np.testing.assert_array_almost_equal(state_ds["air_temperature"].values, forcing_ds["air_temperature"].values)
#         np.testing.assert_array_almost_equal(state_ds["specific_humidity"].values, forcing_ds["specific_humidity"].values)
#         state_ds["air_temperature"] += (forcing_ds["nQ1"] + forcing_ds["pQ1"]) * timestep_seconds
#         state_ds["specific_humidity"] += (forcing_ds["nQ2"] + forcing_ds["pQ1"]) * timestep_seconds


def test_datasets_integrate():
    timestep_seconds = 3 * 60 * 60
    with open("dataset/window_00000.nc", "rb") as f:
        ds = xr.open_dataset(f)

    state_ds = ds.isel(time=0)
    n_timesteps = len(ds["time"])
    for i in range(n_timesteps):
        print(f"Step {i+1} of {n_timesteps}")
        forcing_ds = ds.isel(time=i)
        # state at start of timestep should match forcing dataset state
        np.testing.assert_array_almost_equal(state_ds["air_temperature"].values, forcing_ds["air_temperature"].values)
        np.testing.assert_array_almost_equal(state_ds["specific_humidity"].values, forcing_ds["specific_humidity"].values)
        assert np.sum(np.isnan(forcing_ds["air_temperature_tendency_due_to_model"].values)) == 0
        assert np.sum(np.isnan(forcing_ds["air_temperature_tendency_due_to_nudging"].values)) == 0
        state_ds["air_temperature"].values[:] += (forcing_ds["air_temperature_tendency_due_to_model"].values + forcing_ds["air_temperature_tendency_due_to_nudging"].values) * timestep_seconds
        state_ds["specific_humidity"].values[:] += (forcing_ds["specific_humidity_tendency_due_to_model"].values + forcing_ds["specific_humidity_tendency_due_to_nudging"].values) * timestep_seconds

if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(1)
    test_datasets_integrate()
