from collections import defaultdict
import dataclasses
from matplotlib import pyplot as plt
from typing import Mapping, Optional, List, Tuple, Sequence
import xarray as xr
import tensorflow as tf
from runtime.diagnostics.tensorboard import plot_to_image

U = "eastward_wind"
V = "northward_wind"
T = "air_temperature"
Q = "specific_humidity"


State = Mapping[str, xr.DataArray]


def average(metrics):
    return {key: sum(metrics[key]) / len(metrics[key]) for key in metrics}


@dataclasses.dataclass
class BatchDataConfig:
    training_path: str
    testing_path: str


@dataclasses.dataclass
class OnlineEmulatorConfig:
    """
    Attrs:
        batch_size: the largest batch size to be used for a single gradient
            descent step
        learning_rate: the learning rate for each gradient descent step
        online: whether predictions are used or not
        batch: if provided then these data are used for training the ML model

    """

    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.5
    online: bool = False
    extra_input_variables: List[str] = dataclasses.field(default_factory=list)
    q_weight: float = 1e6
    u_weight: float = 100
    t_weight: float = 100
    v_weight: float = 100
    epochs: int = 1
    batch: Optional[BatchDataConfig] = None
    output_path: str = ""

    @property
    def input_variables(self) -> List[str]:
        return [U, V, T, Q] + list(self.extra_input_variables)

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
        self._statein: Optional[State] = None
        self.output_variables: Sequence[str] = (U, V, T, Q)
        self._step = 0

    @property
    def input_variables(self):
        return self.config.input_variables

    def step(self, in_, out):

        with tf.GradientTape() as tape:
            loss, info = self.get_loss(in_, out)

        vars = self.model.trainable_variables
        grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))

        for key in info:
            tf.summary.scalar(key, info[key], step=self._step)
        self._step += 1
        return info

    def get_loss(self, in_, out):
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
        return (
            loss,
            {
                "loss_u": loss_u.numpy(),
                "loss_v": loss_v.numpy(),
                "loss_q": loss_q.numpy(),
                "loss_t": loss_t.numpy(),
                "loss": loss.numpy(),
            },
        )

    def score(self, d: tf.data.Dataset):
        losses = defaultdict(list)
        for x, y in d.batch(10_000):
            _, info = self.get_loss(x, y)
            for key in info:
                losses[key].append(info[key])
        loss_epoch_test = average(losses)
        return loss_epoch_test

    def batch_fit(self, d: tf.data.Dataset, validation_data=None):
        """

        Args:
            d: a unbatched dataset of tensors. batching is controlled by the
                routines in side this function
            validation_data: an unbatched validation dataset
        """

        if not self.model:
            argsin, argsout = next(iter(d.batch(10_000)))
            self.model = get_model(self.config, argsin, argsout)

        for i in range(self.config.epochs):
            train_loss = defaultdict(lambda: [])
            for x, y in d.batch(self.config.batch_size):
                info = self.step(x, y)
                for key in info:
                    train_loss[key].append(info[key])

            loss_epoch_train = average(train_loss)
            self.log_dict("train_epoch", loss_epoch_train, step=i)

            if validation_data:
                loss_epoch_test = self.score(validation_data)
                self.log_dict("test_epoch", loss_epoch_test, step=i)
                x, y = next(iter(validation_data.batch(3).take(1)))
                out = self.model(x)
                self.log_profiles(
                    "eastward_wind_truth", (y[0] - x[0]).numpy().T, step=i
                )
                self.log_profiles(
                    "eastward_wind_prediction", (out[0] - x[0]).numpy().T, step=i
                )
                self.log_profiles("humidity_truth", (y[3] - x[3]).numpy().T, step=i)
                self.log_profiles(
                    "humidity_prediction", (out[3] - x[3]).numpy().T, step=i
                )

    def log_profiles(self, key, data, step):
        fig = plt.figure()
        plt.plot(data)
        tf.summary.image(key, plot_to_image(fig), step)

    @staticmethod
    def log_dict(prefix, metrics, step):
        for key in metrics:
            tf.summary.scalar(prefix + "/" + key, metrics[key], step=step)

    def partial_fit(self, statein: State, stateout: State):

        in_tensors = _xarray_to_tensor(statein, self.input_variables)
        out_tensors = _xarray_to_tensor(stateout, self.output_variables)
        d = tf.data.Dataset.from_tensor_slices((in_tensors, out_tensors)).shuffle(
            1_000_000
        )
        self.batch_fit(d)

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


def get_model(
    config: OnlineEmulatorConfig, statein: Sequence[tf.Tensor], stateout
) -> tf.keras.Model:
    n = statein[0].shape[-1]
    model = UVTQSimple(n, n, n, n)
    model.fit_scalers(statein, stateout)
    return model
