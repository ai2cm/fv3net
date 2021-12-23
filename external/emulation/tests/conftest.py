import contextlib
import os
import pytest
import tensorflow as tf


@contextlib.contextmanager
def set_current_directory(path):

    origin = os.getcwd()

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def save_input_nml(path):

    input_nml_content = """
    &coupler_nml
        calendar = 'julian'
        current_date = 2016, 8, 1, 0, 0, 0
        dt_atmos = 900
        days = 0
        hours = 1
        minutes = 0
        months = 0
        seconds = 0
    /

    &fv_core_nml
        layout = 1,1
    /
    """

    with open(path, "w") as f:
        f.write(input_nml_content)


@pytest.fixture
def dummy_rundir(tmp_path):

    rundir = tmp_path
    with set_current_directory(rundir):
        save_input_nml(rundir / "input.nml")

        yield rundir


T_in = "air_temperature_input"
T_out = "air_temperature_dummy"
nz = 63


def _create_model():
    in_ = tf.keras.layers.Input(shape=(nz,), name=T_in)
    out_ = tf.keras.layers.Lambda(lambda x: x + 1, name=T_out)(in_)
    model = tf.keras.Model(inputs=in_, outputs=out_)

    return model


def _create_model_dict():
    in_ = tf.keras.layers.Input(shape=(nz,), name=T_in)
    out_ = tf.keras.layers.Lambda(lambda x: x + 1)(in_)
    model = tf.keras.Model(inputs=in_, outputs={T_out: out_})
    return model


def _create_custom_model():
    class CustomModel(tf.keras.Model):
        def call(self, x):
            return {T_out: x[T_in] + 1}

    model = CustomModel()
    d = {T_in: tf.ones((1, nz))}
    model(d)
    return model


@pytest.fixture(
    scope="session", params=[_create_model, _create_model_dict, _create_custom_model]
)
def saved_model_path(tmp_path_factory, request):
    model = request.param()
    path = tmp_path_factory.mktemp("tf_models") / "dummy_model.tf"
    model.save(path.as_posix(), save_format="tf")
    return str(path)
