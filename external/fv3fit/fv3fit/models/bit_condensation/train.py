from fv3fit._shared import io
import tensorflow as tf
from typing import Optional
import numpy as np
from dataclasses import dataclass

from fv3fit._shared.training_config import Hyperparameters, register_training_function
from fv3fit._shared.predictor import Reloadable
from fv3fit.models.bit_condensation import integer_encoding


out_v = "specific_humidity_after_gscond"


# IEEE encoding
def encode_ieee(x):
    buf = x.astype(np.float16).tobytes()
    byte_array = np.frombuffer(buf, dtype=np.uint8)
    n = np.unpackbits(byte_array, axis=-1)
    shape = x.shape + (-1,)
    return n.reshape(shape)

    # IEEE encoding


def decode_ieee(y):
    uint8 = np.packbits(y, -1)
    buf = uint8.tobytes()
    return np.frombuffer(buf, np.float16)


# tests
def test_ieee():
    x = np.array([9.83839], dtype=np.float16)
    x_back = decode_ieee(encode_ieee(x))
    np.testing.assert_array_equal(x_back, x)


assert encode_ieee(np.array([1.0])).shape == (1, 16)


@dataclass
class Config(Hyperparameters):
    depth: int = 2
    width: int = 256
    batch_size: int = 2 ** 14
    epochs: int = 1
    overfit: bool = False

    @property
    def variables(self):
        return set(get_inputs()) | {out_v}


def get_inputs():
    return [
        "air_pressure",
        "surface_air_pressure",
        "surface_air_pressure_after_last_gscond",
        "specific_humidity_input",
        "specific_humidity_after_last_gscond",
        "air_temperature_input",
        "air_temperature_after_last_gscond",
        "pressure_thickness_of_atmospheric_layer",
        "cloud_water_mixing_ratio_input",
    ]


def stack(d):
    p = d["air_pressure"]
    ps = d["surface_air_pressure"][..., None]
    ps_last = d["surface_air_pressure_after_last_gscond"][..., None]
    p_last = p / ps * ps_last
    cloud = tf.math.maximum(1e-20, d["cloud_water_mixing_ratio_input"])
    return tf.stack(
        [
            d["specific_humidity_input"],
            tf.math.log(d["specific_humidity_input"]),
            d["specific_humidity_after_last_gscond"],
            d["air_temperature_input"],
            d["air_temperature_after_last_gscond"],
            p,
            p_last,
            d["pressure_thickness_of_atmospheric_layer"],
            cloud,
            tf.math.log(cloud),
        ],
        -1,
    )


def get_outputs(d):
    return d["specific_humidity_after_gscond"] - d["specific_humidity_input"]


def preprocess(d):
    inputs = stack(d)
    output = get_outputs(d)
    return inputs, output


def encode(x):
    in_bits = tf.py_function(lambda x: encode_ieee(x.numpy()), [x], tf.bool)
    return tf.ensure_shape(in_bits, list(x.shape) + [16])


def decode(y):
    out = tf.py_function(lambda y: decode_ieee(y.numpy()), [y], tf.float16)
    return tf.ensure_shape(out, y.shape[:-1])


def encode_xy(x, y):
    return encode(x), encode(y)


@io.register("bit-condensation")
class KerasModel(Reloadable):
    def __init__(self, model):
        self.model = model

    def dump(self, path: str) -> None:
        """Serialize to a directory."""
        self.model.save(path + "/model.tf")

    @staticmethod
    def load(path: str) -> "KerasModel":
        model = tf.keras.models.load_model(path + "/model.tf")
        return KerasModel(model)

    def predict_numpy(self, d):
        stacked = stack(d).numpy()
        encoded = encode_ieee(stacked)
        encoded_flat = tf.reshape(encoded, [-1, 8, 16])
        y_bits = self.model.predict(encoded_flat) > 0
        y = decode_ieee(y_bits)
        return y.reshape([-1, 79])


class EndPoint(tf.keras.layers.Layer):
    def __init__(self, coder):
        super().__init__()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.coder = coder
        self.dense = tf.keras.layers.Dense(coder.size)

    def call(self, hidden, targets=None):
        logits = tf.squeeze(self.dense(hidden))
        if targets is not None:
            bits = self.coder.encode(targets)
            y_hat = self.coder.decode(logits)

            y_hat = tf.cast(y_hat, tf.float32)
            targets = tf.cast(targets, tf.float32)

            target_decoded = self.coder.decode(self.coder.encode(targets))

            mse = tf.keras.metrics.mean_squared_error(targets, y_hat)
            mse_null = tf.keras.metrics.mean_squared_error(
                targets, tf.zeros_like(targets)
            )
            mse_coding = tf.keras.metrics.mean_squared_error(targets, target_decoded)
            r2 = 1 - mse / mse_null
            self.add_metric(mse, aggregation="mean", name="mse")
            self.add_metric(mse_coding, aggregation="mean", name="mse_coding")
            self.add_metric(mse_null, aggregation="mean", name="mse_null")
            self.add_metric(r2, aggregation="mean", name="r2")
            loss = self.coder.loss(bits, logits)
            self.add_loss(loss)
        return tf.nn.sigmoid(logits)


@register_training_function("bit_condensation", Config)
def train(
    hyperparameters: Config,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> KerasModel:
    b = train_batches.map(preprocess)

    grid_points = b.unbatch().unbatch()

    if hyperparameters.overfit:
        grid_points = grid_points.take(10)
    else:
        grid_points = grid_points.shuffle(10_000)

    grid_points = grid_points.batch(hyperparameters.batch_size).map(
        lambda x, y: {"x": x, "y": y}
    )
    input_norm = tf.keras.layers.Normalization(axis=1)
    input_norm.adapt(grid_points.map(lambda x: x["x"]).take(10))

    x = tf.keras.Input(10, name="x")
    y = tf.keras.Input([], name="y")
    x_norm = input_norm(x)

    state = x_norm
    for i in range(hyperparameters.depth):
        dense = tf.keras.layers.Dense(hyperparameters.width, activation="relu")
        state = dense(state)

    # endpoint pattern
    # https://keras.io/examples/keras_recipes/endpoint_layer_pattern/
    dt = tf.int16

    coder = integer_encoding.IntEncoder(1e-5, dt=dt)
    coder = integer_encoding.OneHot(1e-3)
    coder = integer_encoding.Float()
    coder = integer_encoding.Mixed()
    endpoint = EndPoint(coder)
    preds = endpoint(state, y)
    model = tf.keras.Model([x, y], preds)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    model.fit(grid_points, epochs=hyperparameters.epochs)
    return KerasModel(model)
