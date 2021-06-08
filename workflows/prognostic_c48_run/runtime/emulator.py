import dataclasses
from typing import Mapping, Optional, Sequence, Tuple
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
    extra_input_variables: Sequence[str] = ()
    q_weight: float = 1e6
    u_weight: float = 100
    t_weight: float = 100
    v_weight: float = 100

    @property
    def input_variables(self) -> Tuple[str]:
        return (U, V, T, Q) + tuple(self.extra_input_variables)


def stack(state: State, keys) -> xr.Dataset:
    ds = xr.Dataset({key: state[key] for key in keys})
    sample_dims = ["y", "x"]
    return ds.stack(sample=sample_dims).transpose("sample", ...)


def to_tensor(arr: xr.DataArray) -> tf.Variable:
    return tf.cast(tf.Variable(arr), tf.float32)


def to_tensors(ds: xr.Dataset, keys) -> Tuple[xr.DataArray]:
    return tuple([to_tensor(ds[k]) for k in ds])


def _xarray_to_tensor(state, keys):
    in_ = stack(state, keys)
    return to_tensors(in_, keys)


class OnlineEmulator:
    def __init__(
        self, config: OnlineEmulatorConfig,
    ):
        self.config = config
        self.model = None
        self.optimizer = tf.optimizers.SGD(
            learning_rate=config.learning_rate, momentum=config.momentum
        )
        self.scaler_fitted: bool = False
        self._statein: Optional[State] = None
        self.output_variables: Sequence[str] = (U, V, T, Q)

    @property
    def input_variables(self):
        return self.config.input_variables

    def partial_fit(self, statein: State, stateout: State):

        if self.model is None:
            self.model = get_model(self.config, statein)
            self.step = 0

        def step(self, in_, out):
            with tf.GradientTape() as tape:

                up, vp, tp, qp = self.model(in_)
                ut, vt, tt, qt = out
                loss_u = tf.reduce_mean(tf.keras.losses.mean_squared_error(ut, up))
                loss_v = tf.reduce_mean(tf.keras.losses.mean_squared_error(vt, vp))
                loss_t = tf.reduce_mean(tf.keras.losses.mean_squared_error(tt, tp))
                loss_q = tf.reduce_mean(tf.keras.losses.mean_squared_error(qt, qp))
                loss = (
                    loss_u * self.config.u_weight
                    + loss_v * self.config.v_weight
                    + loss_t * self.config.t_weight
                    + loss_q * self.config.q_weight
                )

            vars = self.model.trainable_variables
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))

            tf.summary.scalar("loss_u", loss_u, step=self.step)
            tf.summary.scalar("loss_v", loss_v, step=self.step)
            tf.summary.scalar("loss_t", loss_t, step=self.step)
            tf.summary.scalar("loss_q", loss_q, step=self.step)
            tf.summary.scalar("loss", loss, step=self.step)
            self.step += 1

        in_tensors = _xarray_to_tensor(statein, self.input_variables)
        out_tensors = _xarray_to_tensor(stateout, self.output_variables)
        d = tf.data.Dataset.from_tensor_slices((in_tensors, out_tensors))

        if self.model is not None:
            if not self.scaler_fitted:
                self.scaler_fitted = True
                argsin, argsout = next(iter(d.batch(len(d))))
                self.model.fit_scalers(argsin, argsout)
            for x, y in d.shuffle(1_000_000).batch(self.config.batch_size):
                step(self, x, y)

    def predict(self, state: State) -> State:
        if self.model is None:
            raise ValueError("Must call .partial_fit at least once.")

        in_ = stack(state, self.input_variables)
        up, vp, tp, qp = self.model(to_tensors(in_, self.input_variables))
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


class ScalarNormLayer(NormLayer):
    """UnNormalize a vector using a scalar mean and standard deviation


    """

    def __init__(self, name=None):
        super(ScalarNormLayer, self).__init__(name=name)

    def build(self, in_shape):
        self.mean = self.add_weight(
            "mean", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )
        self.sigma = self.add_weight(
            "sigma", shape=[], dtype=tf.float32, trainable=False
        )

    def fit(self, tensor):
        self(tensor)
        self.mean.assign(tf.cast(tf.reduce_mean(tensor, axis=0), tf.float32))
        self.sigma.assign(
            tf.cast(tf.sqrt(tf.reduce_mean((tensor - self.mean) ** 2)), tf.float32,)
        )

    def call(self, tensor):
        return tensor * self.sigma + self.mean


def atleast_2d(x: tf.Variable) -> tf.Variable:
    n = len(x.shape)
    if n == 1:
        return tf.reshape(x, shape=x.shape + [1])
    else:
        return x


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

        self.scalers = [ScalarNormLayer(name=f"out_{i}") for i in range(4)]

    def _fit_input_scaler(self, args: Sequence[tf.Variable]):
        args = [atleast_2d(arg) for arg in args]
        stacked = tf.concat(args, axis=-1)
        self.norm.fit(stacked)

    def _fit_output_scaler(
        self, argsin: Sequence[tf.Variable], argsout: Sequence[tf.Variable]
    ):
        for i in range(len(self.scalers)):
            self.scalers[i].fit(argsout[i] - argsin[i])

    def fit_scalers(
        self, argsin: Sequence[tf.Variable], argsout: Sequence[tf.Variable]
    ):
        self._fit_input_scaler(argsin)
        self._fit_output_scaler(argsin, argsout)

    def call(self, args: Sequence[tf.Variable]):
        # assume has dims: batch, z
        u, v, t, q = args[:4]
        args = [atleast_2d(arg) for arg in args]
        stacked = tf.concat(args, axis=-1)
        hidden = self.relu(self.linear(self.norm(stacked)))

        return (
            u + self.scalers[0](self.out_u(hidden)),
            v + self.scalers[1](self.out_v(hidden)),
            t + self.scalers[2](self.out_t(hidden)),
            q + self.scalers[3](self.out_q(hidden)),
        )


def needs_restart(state) -> bool:
    """Detect if error state is happening, in which case we should restart the
    model from a clean state
    """
    return False


def get_model(config: OnlineEmulatorConfig, state: State) -> tf.keras.Model:
    n = state["eastward_wind"].sizes["z"]
    return UVTQSimple(n, n, n, n)
