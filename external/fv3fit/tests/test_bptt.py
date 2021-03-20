import pytest
import xarray as xr
import numpy as np
from datetime import timedelta
from cftime import DatetimeProlepticGregorian as datetime
from fv3fit.keras._models.recurrent import BPTTModel


@pytest.fixture
def sample_dim_name():
    return "sample"

@pytest.fixture
def dt():
    return timedelta(hours=1)


@pytest.fixture
def train_dataset(sample_dim_name, dt):
    n_samples = 8
    n_times = 5
    n_z = 12
    t_start = datetime(2020, 1, 1)
    ds = xr.Dataset(
        data_vars={
            "air_temperature": xr.DataArray(
                np.random.randn(n_samples, n_times, n_z),
                dims=[sample_dim_name, "time", "z"],
                attrs={"units": "K"}
            ),
            "specific_humidity": xr.DataArray(
                np.random.randn(n_samples, n_times, n_z),
                dims=[sample_dim_name, "time", "z"],
                attrs={"units": "kg/kg"}
            ),
            "air_temperature_tendency_due_to_model": xr.DataArray(
                np.zeros([n_samples, n_times, n_z]),
                dims=[sample_dim_name, "time", "z"],
            ),
            "specific_humidity_tendency_due_to_model": xr.DataArray(
                np.zeros([n_samples, n_times, n_z]),
                dims=[sample_dim_name, "time", "z"],
            ),
            "a": xr.DataArray(
                np.random.randn(n_samples, n_times),
                dims=[sample_dim_name, "time"],
            ),
            "b": xr.DataArray(
                np.random.randn(n_samples, n_times),
                dims=[sample_dim_name, "time"],
            ),
        },
        coords = {
            "time": np.asarray([t_start + i * dt for i in range(n_times)]),
        }
    )
    return ds

def test_predict_model_gives_tendency_of_train_model(train_dataset, sample_dim_name, dt):
    model = BPTTModel(
        sample_dim_name,
        ["a", "b"],
        n_units=32,
        n_hidden_layers=4,
        kernel_regularizer=None,
        train_batch_size=48,
        optimizer="adam",
    )
    model.build_for(train_dataset)

    train_dataset["air_temperature_tendency_due_to_model"][:] = 0.
    train_dataset["specific_humidity_tendency_due_to_model"][:] = 0.

    air_temperature, specific_humidity = model.train_model.predict(
        x=model.get_keras_inputs(train_dataset)
    )

    tendency_ds = model.predictor_model.predict(train_dataset.isel(time=0))

    train_tendency = (air_temperature[:, 1, :] - air_temperature[:, 0, :]) / dt.total_seconds()
    np.testing.assert_array_almost_equal(train_tendency, tendency_ds["dQ1"].values)

    train_tendency = (specific_humidity[:, 1, :] - specific_humidity[:, 0, :]) / dt.total_seconds()
    np.testing.assert_array_almost_equal(train_tendency, tendency_ds["dQ2"].values)


def test_predict_model_gives_different_tendencies(train_dataset, sample_dim_name, dt):
    model = BPTTModel(
        sample_dim_name,
        ["a", "b"],
        n_units=32,
        n_hidden_layers=4,
        kernel_regularizer=None,
        train_batch_size=48,
        optimizer="adam",
    )
    model.build_for(train_dataset)

    tendency_ds = model.predictor_model.predict(train_dataset.isel(time=0))
    assert not np.all(tendency_ds["dQ1"].values == tendency_ds["dQ2"].values)


def test_train_model_gives_different_outputs(train_dataset, sample_dim_name, dt):
    model = BPTTModel(
        sample_dim_name,
        ["a", "b"],
        n_units=32,
        n_hidden_layers=4,
        kernel_regularizer=None,
        train_batch_size=48,
        optimizer="adam",
    )
    model.build_for(train_dataset)

    air_temperature, specific_humidity = model.train_model.predict(
        x=model.get_keras_inputs(train_dataset)
    )
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 1)
    # print(air_temperature.shape)
    # ax[0].pcolormesh(air_temperature[0, :] - air_temperature[0, 0, :])
    # ax[1].pcolormesh(specific_humidity[0, :] - air_temperature[0, 0, :])
    # plt.tight_layout()
    # plt.show()
    assert not np.all(air_temperature == specific_humidity)
