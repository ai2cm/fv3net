import pytest
import xarray as xr
import tensorflow as tf
import numpy as np
from datetime import timedelta
from cftime import DatetimeProlepticGregorian as datetime
from fv3fit.keras._models.recurrent import _BPTTModel as BPTTModel
import tempfile
import fv3fit
import os
import subprocess


@pytest.fixture
def sample_dim_name():
    return "sample"


@pytest.fixture
def dt():
    return timedelta(seconds=1)


@pytest.fixture(params=["limiter", "no_limiter"])
def use_moisture_limiter(request):
    if request.param == "limiter":
        return True
    elif request.param == "no_limiter":
        return False
    else:
        raise NotImplementedError(request.param)


def get_train_dataset(sample_dim_name, dt):
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


def get_model(sample_dim_name, use_moisture_limiter):
    return BPTTModel(
        sample_dim_name,
        ["a", "b"],
        n_units=32,
        n_hidden_layers=4,
        kernel_regularizer=None,
        train_batch_size=48,
        optimizer="adam",
        use_moisture_limiter=use_moisture_limiter,
    )


def test_predict_model_gives_tendency_of_train_model(
    sample_dim_name, dt, use_moisture_limiter
):
    np.random.seed(0)
    tf.random.set_seed(1)
    train_dataset = get_train_dataset(sample_dim_name, dt)
    model = get_model(sample_dim_name, use_moisture_limiter)
    model.fit_statistics(train_dataset)
    model.fit(train_dataset)

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


def test_fit_bptt_command_line(sample_dim_name, dt):
    config_text = """
regularizer:
  name: l2
  kwargs:
    l: 0.0001
optimizer:
  name: Adam
  kwargs:
    learning_rate: 0.001
    amsgrad: True
    clipnorm: 1.0
hyperparameters:
  n_hidden_layers: 3
  n_units: 256
  train_batch_size: 48
  use_moisture_limiter: true
  state_noise: 0.25
input_variables:
  - a
  - b
decrease_learning_rate_epoch: 1
decreased_learning_rate: 0.0001
total_epochs: 2
random_seed: 0
"""
    training_dataset = get_train_dataset(sample_dim_name, dt)
    with tempfile.TemporaryDirectory() as tmpdir:
        arrays_dir = os.path.join(tmpdir, "arrays")
        os.mkdir(arrays_dir)
        for i in range(3):
            filename = os.path.join(arrays_dir, f"arrays_{i}.nc")
            training_dataset.to_netcdf(filename)
        config_filename = os.path.join(tmpdir, "config.yaml")
        with open(config_filename, "w") as f:
            f.write(config_text)
        outdir = os.path.join(tmpdir, "output")
        subprocess.check_call(
            ["python", "-m", "fv3fit.train_bptt", arrays_dir, config_filename, outdir]
        )
        # as a minimum, test that output exists
        assert os.path.isdir(outdir)
        assert len(os.listdir(outdir)) > 0

        # one model per epoch is saved
        for sample_model_dir in os.listdir(outdir):
            loaded = fv3fit.load(os.path.join(outdir, sample_model_dir))

            first_timestep = training_dataset.isel(time=0)
            loaded_output = loaded.predict(first_timestep)
            assert isinstance(loaded_output, xr.Dataset)


def test_predict_model_can_predict_columnwise(sample_dim_name, dt):
    train_dataset = get_train_dataset(sample_dim_name, dt)
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
    model.fit_statistics(train_dataset)
    model.fit(train_dataset)

    assert sample_dim_name != "different_sample_dim_name"
    test_dataset = train_dataset.rename({sample_dim_name: "different_sample_dim_name"})

    model.predictor_model.predict_columnwise(test_dataset.isel(time=0), feature_dim="z")


def test_train_model_uses_correct_given_tendency(sample_dim_name, dt):
    train_dataset = get_train_dataset(sample_dim_name, dt)
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
    model.fit_statistics(train_dataset)
    model.fit(train_dataset)

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


def test_predict_model_gives_different_tendencies(sample_dim_name, dt):
    train_dataset = get_train_dataset(sample_dim_name, dt)
    model = BPTTModel(
        sample_dim_name,
        ["a", "b"],
        n_units=32,
        n_hidden_layers=4,
        kernel_regularizer=None,
        train_batch_size=48,
        optimizer="adam",
    )
    model.fit_statistics(train_dataset)
    model.fit(train_dataset)

    tendency_ds = model.predictor_model.predict(train_dataset.isel(time=0))
    assert not np.all(tendency_ds["dQ1"].values == tendency_ds["dQ2"].values)


def test_train_model_gives_different_outputs(sample_dim_name, dt):
    train_dataset = get_train_dataset(sample_dim_name, dt)
    model = BPTTModel(
        sample_dim_name,
        ["a", "b"],
        n_units=32,
        n_hidden_layers=4,
        kernel_regularizer=None,
        train_batch_size=48,
        optimizer="adam",
    )
    model.fit_statistics(train_dataset)
    model.fit(train_dataset)

    air_temperature, specific_humidity = model.train_keras_model.predict(
        x=model.get_keras_inputs(train_dataset)
    )
    assert not np.all(air_temperature == specific_humidity)


def test_integrate_stepwise(sample_dim_name, dt):
    """
    We need an integrate_stepwise routine for scripts that evaluate the model
    timeseries, since the saved model operates on a single timestep. This code
    is hard to write for a variety of reasons.
    
    The test here makes sure that the integration code performs an identical
    integration to the one used to train the model.
    
    train_keras_model does this by predicting a timeseries
    directly, while integrate_stepwise integrates the stepwise model over
    many timesteps.
    """
    train_dataset = get_train_dataset(sample_dim_name, dt)
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
    model.fit_statistics(train_dataset)
    model.fit(train_dataset)

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


def test_reloaded_model_gives_same_outputs(sample_dim_name, dt):
    train_dataset = get_train_dataset(sample_dim_name, dt)
    model = BPTTModel(
        sample_dim_name,
        ["a", "b"],
        n_units=32,
        n_hidden_layers=4,
        kernel_regularizer=None,
        train_batch_size=48,
        optimizer="adam",
    )
    model.fit_statistics(train_dataset)
    model.fit(train_dataset)

    with tempfile.TemporaryDirectory() as tmpdir:
        fv3fit.dump(model.predictor_model, tmpdir)
        loaded = fv3fit.load(tmpdir)

    first_timestep = train_dataset.isel(time=0)
    reference_output = model.predictor_model.predict(first_timestep)
    # test that loaded model gives the same predictions
    loaded_output = loaded.predict(first_timestep)
    xr.testing.assert_equal(reference_output, loaded_output)
