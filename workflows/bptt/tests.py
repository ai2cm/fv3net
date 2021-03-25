import xarray as xr
import numpy as np
import tensorflow as tf


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
        np.testing.assert_array_almost_equal(
            state_ds["air_temperature"].values, forcing_ds["air_temperature"].values
        )
        np.testing.assert_array_almost_equal(
            state_ds["specific_humidity"].values, forcing_ds["specific_humidity"].values
        )
        assert (
            np.sum(np.isnan(forcing_ds["air_temperature_tendency_due_to_model"].values))
            == 0
        )
        assert (
            np.sum(
                np.isnan(forcing_ds["air_temperature_tendency_due_to_nudging"].values)
            )
            == 0
        )
        state_ds["air_temperature"].values[:] += (
            forcing_ds["air_temperature_tendency_due_to_model"].values
            + forcing_ds["air_temperature_tendency_due_to_nudging"].values
        ) * timestep_seconds
        state_ds["specific_humidity"].values[:] += (
            forcing_ds["specific_humidity_tendency_due_to_model"].values
            + forcing_ds["specific_humidity_tendency_due_to_nudging"].values
        ) * timestep_seconds


if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_seed(1)
    test_datasets_integrate()
