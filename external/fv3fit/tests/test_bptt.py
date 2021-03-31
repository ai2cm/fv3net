import pytest
import xarray as xr
import tensorflow as tf
import numpy as np
from datetime import timedelta
from cftime import DatetimeProlepticGregorian as datetime
from fv3fit.keras._models.recurrent import BPTTModel
import tempfile
import fv3fit


@pytest.fixture
def sample_dim_name():
    return "sample"


@pytest.fixture
def dt():
    return timedelta(seconds=1)


@pytest.fixture
def train_dataset(sample_dim_name, dt):
    n_samples = 6
    n_times = 12
    n_z = 16
    t_start = datetime(2020, 1, 1)
    ds = xr.Dataset(
        data_vars={
            "air_temperature": xr.DataArray(
                np.random.randn(n_samples, n_times, n_z),
                dims=[sample_dim_name, "time", "z"],
                attrs={"units": "K"},
            ),
            "specific_humidity": xr.DataArray(
                np.random.randn(n_samples, n_times, n_z),
                dims=[sample_dim_name, "time", "z"],
                attrs={"units": "kg/kg"},
            ),
            "air_temperature_tendency_due_to_model": xr.DataArray(
                np.random.randn(n_samples, n_times, n_z),
                dims=[sample_dim_name, "time", "z"],
            ),
            "specific_humidity_tendency_due_to_model": xr.DataArray(
                np.random.randn(n_samples, n_times, n_z),
                dims=[sample_dim_name, "time", "z"],
            ),
            "air_temperature_tendency_due_to_nudging": xr.DataArray(
                np.random.randn(n_samples, n_times, n_z),
                dims=[sample_dim_name, "time", "z"],
            ),
            "specific_humidity_tendency_due_to_nudging": xr.DataArray(
                np.random.randn(n_samples, n_times, n_z),
                dims=[sample_dim_name, "time", "z"],
            ),
            "a": xr.DataArray(
                np.random.randn(n_samples, n_times), dims=[sample_dim_name, "time"],
            ),
            "b": xr.DataArray(
                np.random.randn(n_samples, n_times), dims=[sample_dim_name, "time"],
            ),
        },
        coords={"time": np.asarray([t_start + i * dt for i in range(n_times)])},
    )
    return ds


def test_predict_model_gives_tendency_of_train_model(
    train_dataset, sample_dim_name, dt
):
    np.random.seed(0)
    tf.random.set_seed(1)
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

    train_dataset["air_temperature_tendency_due_to_model"][:] = 0.0
    train_dataset["specific_humidity_tendency_due_to_model"][:] = 0.0

    air_temperature, specific_humidity = model.train_keras_model.predict(
        x=model.get_keras_inputs(train_dataset)
    )
    dQ1, dQ2 = model.train_tendency_model.predict(
        x=model.get_keras_inputs(train_dataset)
    )

    tendency_ds = model.predictor_model.predict(train_dataset.isel(time=0))

    # take difference of first output from initial input state
    Q1_train_tendency = (
        air_temperature[:, 0, :] - train_dataset["air_temperature"][:, 0, :]
    ) / (dt.total_seconds())
    Q2_train_tendency = (
        specific_humidity[:, 0, :] - train_dataset["specific_humidity"][:, 0, :]
    ) / (dt.total_seconds())

    np.testing.assert_allclose(dQ1[:, 0, :], tendency_ds["dQ1"].values, atol=1e-6)
    np.testing.assert_allclose(dQ2[:, 0, :], tendency_ds["dQ2"].values, atol=1e-6)
    np.testing.assert_allclose(dQ1[:, 0, :], Q1_train_tendency, atol=1e-6)
    np.testing.assert_allclose(dQ2[:, 0, :], Q2_train_tendency, atol=1e-6)


def test_predict_model_can_predict_columnwise(train_dataset, sample_dim_name, dt):
    np.random.seed(0)
    tf.random.set_seed(1)
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

    assert sample_dim_name != "different_sample_dim_name"
    test_dataset = train_dataset.rename({sample_dim_name: "different_sample_dim_name"})

    model.predictor_model.predict_columnwise(test_dataset.isel(time=0), feature_dim="z")


def test_train_model_uses_correct_given_tendency(train_dataset, sample_dim_name, dt):
    np.random.seed(0)
    tf.random.set_seed(1)
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

    assert not np.all(
        train_dataset["air_temperature_tendency_due_to_model"].values == 0.0
    )

    air_temperature, specific_humidity = model.train_keras_model.predict(
        x=model.get_keras_inputs(train_dataset)
    )
    dQ1, dQ2 = model.train_tendency_model.predict(
        x=model.get_keras_inputs(train_dataset)
    )

    # take difference of first output from initial input state
    Q1_train_tendency = (
        air_temperature[:, 0, :] - train_dataset["air_temperature"][:, 0, :]
    ) / (dt.total_seconds())
    Q2_train_tendency = (
        specific_humidity[:, 0, :] - train_dataset["specific_humidity"][:, 0, :]
    ) / (dt.total_seconds())

    np.testing.assert_allclose(
        Q1_train_tendency - dQ1[:, 0, :],
        train_dataset["air_temperature_tendency_due_to_model"][:, 0, :].values,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        Q2_train_tendency - dQ2[:, 0, :],
        train_dataset["specific_humidity_tendency_due_to_model"][:, 0, :].values,
        atol=1e-6,
    )

    # check for second timestep also
    Q1_train_tendency = (air_temperature[:, 1, :] - air_temperature[:, 0, :]) / (
        dt.total_seconds()
    )
    Q2_train_tendency = (specific_humidity[:, 1, :] - specific_humidity[:, 0, :]) / (
        dt.total_seconds()
    )

    np.testing.assert_allclose(
        Q1_train_tendency - dQ1[:, 1, :],
        train_dataset["air_temperature_tendency_due_to_model"][:, 1, :].values,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        Q2_train_tendency - dQ2[:, 1, :],
        train_dataset["specific_humidity_tendency_due_to_model"][:, 1, :].values,
        atol=1e-6,
    )


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


def test_train_model_gives_different_outputs(train_dataset, sample_dim_name):
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

    air_temperature, specific_humidity = model.train_keras_model.predict(
        x=model.get_keras_inputs(train_dataset)
    )
    assert not np.all(air_temperature == specific_humidity)


def test_integrate_stepwise(train_dataset, sample_dim_name):
    np.random.seed(0)
    tf.random.set_seed(1)
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

    air_temperature, specific_humidity = model.train_keras_model.predict(
        x=model.get_keras_inputs(train_dataset)
    )
    out_ds = model.predictor_model.integrate_stepwise(train_dataset)
    np.testing.assert_allclose(
        out_ds["air_temperature"].values[:, 0, :], air_temperature[:, 0, :], atol=1e-6
    )
    np.testing.assert_allclose(
        out_ds["specific_humidity"].values[:, 0, :],
        specific_humidity[:, 0, :],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out_ds["air_temperature"].values[:, 1, :], air_temperature[:, 1, :], atol=1e-6
    )
    np.testing.assert_allclose(
        out_ds["specific_humidity"].values[:, 1, :],
        specific_humidity[:, 1, :],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out_ds["air_temperature"].values, air_temperature, atol=1e-5
    )
    np.testing.assert_allclose(
        out_ds["specific_humidity"].values, specific_humidity, atol=1e-5
    )
    assert not np.all(air_temperature == specific_humidity)


def test_reloaded_model_gives_same_outputs(train_dataset, sample_dim_name):
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

    with tempfile.TemporaryDirectory() as tmpdir:
        fv3fit.dump(model.predictor_model, tmpdir)
        loaded = fv3fit.load(tmpdir)

    first_timestep = train_dataset.isel(time=0)
    reference_output = model.predictor_model.predict(first_timestep)
    # test that loaded model gives the same predictions
    loaded_output = loaded.predict(first_timestep)
    for name in reference_output.data_vars.keys():
        xr.testing.assert_equal(reference_output[name], loaded_output[name])
