import dataclasses
from typing import Mapping, Optional
import xarray as xr
import tensorflow as tf

U = "eastward_wind"
V = "northward_wind"
T = "air_temperature"
Q = "specific_humidity"


State = Mapping[str, xr.DataArray]


@dataclasses.dataclass
class OnlineEmulatorConfig:
    """
    Attrs:
        batch_size: the largest batch size to be used for a single gradient
            descent step
        learning_rate: the learning rate for each gradient descent step
        online: whether predictions are used or not

    """

    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.5
    online: bool = False


def stack(state: State, keys) -> xr.Dataset:
    ds = xr.Dataset({key: state[key] for key in keys})
    sample_dims = ["y", "x"]
    return ds.stack(sample=sample_dims).transpose("sample", ...)


def to_tensor(arr: xr.DataArray) -> tf.Variable:
    return tf.cast(tf.Variable(arr), tf.float32)


def to_tensors(ds: xr.Dataset):
    return to_tensor(ds[U]), to_tensor(ds[V]), to_tensor(ds[T]), to_tensor(ds[Q])


KEYS = [U, V, T, Q]


def xarray_to_dataset(statein: State, stateout: State) -> tf.data.Dataset:
    in_ = stack(statein, KEYS)
    out = stack(stateout, KEYS)
    out_tensors = to_tensors(out)
    in_tensors = to_tensors(in_)
    return tf.data.Dataset.from_tensor_slices((out_tensors, in_tensors))


class OnlineEmulator:
    def __init__(self, config: OnlineEmulatorConfig):
        self.config = config
        self.model = None
        self.optimizer = tf.optimizers.SGD(
            learning_rate=config.learning_rate, momentum=config.momentum
        )
        self.scaler_fitted: bool = False
        self._statein: Optional[State] = None

    def set_input_state(self, state: State):
        self._statein = {key: state[key] for key in KEYS}

    def observe_predict(self, state: State) -> State:
        if self._statein is None:
            raise ValueError("Must call `set_input_state` before this routine.")
        else:
            stateout = {key: state[key] for key in KEYS}
            self.partial_fit(self._statein, stateout)
            out = self.predict(self._statein)
            self._statein = None
            return out

    def partial_fit(self, statein: State, stateout: State):

        if self.model is None:
            self.model = get_model(self.config, statein)
            self.step = 0

        def step(self, in_, out):
            with tf.GradientTape() as tape:

                up, vp, tp, qp = self.model(*in_)
                ut, vt, tt, qt = out
                loss_u = tf.reduce_mean(tf.keras.losses.mean_squared_error(ut, up))
                loss_v = tf.reduce_mean(tf.keras.losses.mean_squared_error(vt, vp))
                loss_t = tf.reduce_mean(tf.keras.losses.mean_squared_error(tt, tp))
                loss_q = tf.reduce_mean(tf.keras.losses.mean_squared_error(qt, qp))
                loss = loss_u * 100 + loss_v * 100 + loss_t * 100 + loss_q * 1e6

            vars = self.model.trainable_variables
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))

            tf.summary.scalar("loss_u", loss_u, step=self.step)
            tf.summary.scalar("loss_v", loss_v, step=self.step)
            tf.summary.scalar("loss_t", loss_t, step=self.step)
            tf.summary.scalar("loss_q", loss_q, step=self.step)
            tf.summary.scalar("loss", loss, step=self.step)
            self.step += 1

        d = xarray_to_dataset(statein, stateout)

        if self.model is not None:
            if not self.scaler_fitted:
                self.scaler_fitted = True
                self.model.fit_scaler(*next(iter(d.batch(len(d))))[0])
            for x, y in d.shuffle(1_000_000).batch(self.config.batch_size):
                step(self, x, y)

    def predict(self, state: State) -> State:
        if self.model is None:
            raise ValueError("Must call .partial_fit at least once.")

        keys = [U, V, T, Q]
        in_ = stack(state, keys)
        up, vp, tp, qp = self.model(*to_tensors(in_))
        dims = ["sample", "z"]

        attrs = {"units": "no one cares"}

        return xr.Dataset(
            {
                U: (dims, up, attrs),
                V: (dims, vp, attrs),
                T: (dims, tp, attrs),
                Q: (dims, qp, attrs),
            },
            coords=in_.coords,
        ).unstack("sample")


class NormLayer(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-7, name=None):
        super(NormLayer, self).__init__(name=name)
        self.epsilon = epsilon

    def build(self, in_shape):
        self.mean = self.add_weight(
            "mean", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )
        self.sigma = self.add_weight(
            "sigma", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )
        self.fitted = False

    def fit(self, tensor):
        self(tensor)
        self.mean.assign(tf.cast(tf.reduce_mean(tensor, axis=0), tf.float32))
        self.sigma.assign(
            tf.cast(
                tf.sqrt(tf.reduce_mean((tensor - self.mean) ** 2, axis=0)), tf.float32,
            )
        )

    def call(self, tensor):
        return (tensor - self.mean) / (self.sigma + self.epsilon)


class UVTQSimple(tf.keras.layers.Layer):
    def __init__(self, u_size, v_size, t_size, q_size):
        super(UVTQSimple, self).__init__()
        self.norm = NormLayer(name="norm")
        self.linear = tf.keras.layers.Dense(256, name="lin")
        self.relu = tf.keras.layers.ReLU()
        self.out_u = tf.keras.layers.Dense(u_size, name="out_u")
        self.out_v = tf.keras.layers.Dense(v_size, name="out_v")
        self.out_t = tf.keras.layers.Dense(t_size, name="out_t")
        self.out_q = tf.keras.layers.Dense(q_size, name="out_q")

    def fit_scaler(self, u, v, t, q):
        stacked = tf.concat([u, v, t, q], axis=-1)
        self.norm.fit(stacked)

    def call(self, u, v, t, q):
        # assume has dims: batch, z
        stacked = tf.concat([u, v, t, q], axis=-1)
        hidden = self.relu(self.linear(self.norm(stacked)))

        return (
            u + self.out_u(hidden) / 10,
            v + self.out_v(hidden) / 10,
            t + self.out_t(hidden) / 10,
            q + self.out_q(hidden) / 1000,
        )


def needs_restart(state) -> bool:
    """Detect if error state is happening, in which case we should restart the
    model from a clean state
    """
    return False


def get_model(config: OnlineEmulatorConfig, state: State) -> tf.keras.Model:
    n = state["eastward_wind"].sizes["z"]
    return UVTQSimple(n, n, n, n)
