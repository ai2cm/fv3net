import tensorflow as tf
from pathlib import Path


def create_model():
    in_ = tf.keras.layers.Input(shape=(63,), name="air_temperature_input")
    out_ = tf.keras.layers.Lambda(lambda x: x, name="air_temperature_dummy")(in_)
    model = tf.keras.Model(inputs=in_, outputs=out_)

    return model


if __name__ == "__main__":

    model = create_model()
    path = Path(__file__).parent / "dummy_model.tf"
    model.save(path.as_posix(), save_format="tf")
