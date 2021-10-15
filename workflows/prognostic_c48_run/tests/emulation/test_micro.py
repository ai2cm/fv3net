import pytest
import tensorflow as tf


def create_model():
    in_ = tf.keras.layers.Input(shape=(63,), name="air_temperature_input")
    out_ = tf.keras.layers.Lambda(lambda x: x, name="air_temperature_dummy")(in_)
    model = tf.keras.Model(inputs=in_, outputs=out_)

    return model


@pytest.fixture(scope="session")
def saved_model(tmp_path):
    model = create_model()
    path = tmp_path / "dummy_model.tf"
    model.save(path.as_posix(), save_format="tf")

    return str(path)
