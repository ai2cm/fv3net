from collections import defaultdict
import dataclasses
from typing import Mapping, Optional, Sequence, Tuple, Union, List
import os
from runtime.weights_and_biases import WandBLogger
import xarray as xr
import numpy
import tensorflow as tf
import dacite
from runtime.diagnostics.tensorboard import ConsoleLogger, TBLogger, LoggerList
from runtime.loss import ScalarLoss, MultiVariableLoss
import logging
import json


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
        num_hidden_layers: number of hidden layers used. Only implemented for
            ScalarLoss targets.
        wandb_logger: if True, then enable weights and biases saving

    """

    batch_size: int = 64
    learning_rate: float = 0.01
    num_hidden: int = 256
    num_hidden_layers: int = 1
    momentum: float = 0.5
    online: bool = False
    extra_input_variables: List[str] = dataclasses.field(default_factory=list)
    epochs: int = 1
    levels: int = 79
    batch: Optional[BatchDataConfig] = None
    target: Union[MultiVariableLoss, ScalarLoss] = dataclasses.field(
        default_factory=MultiVariableLoss
    )
    wandb_logger: bool = False

    output_path: str = ""

    @property
    def input_variables(self) -> List[str]:
        return [U, V, T, Q] + list(self.extra_input_variables)

    @classmethod
    def from_dict(cls, dict_) -> "OnlineEmulatorConfig":
        return dacite.from_dict(cls, dict_, dacite.Config(strict=True))


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
        self.model = get_model(config)
        self.optimizer = tf.optimizers.SGD(
            learning_rate=config.learning_rate, momentum=config.momentum
        )
        self._statein: Optional[State] = None
        self.output_variables: Sequence[str] = (U, V, T, Q)
        self._step = 0
        self.logger = LoggerList([TBLogger(), ConsoleLogger(), WandBLogger()])

        self.logger = LoggerList([TBLogger(), ConsoleLogger()])

        if config.wandb_logger:
            self.logger.loggers.append(WandBLogger())

    @property
    def input_variables(self):
        return self.config.input_variables

    def step(self, in_, out):

        with tf.GradientTape() as tape:
            loss, info = self.get_loss(in_, out)

        vars = self.model.trainable_variables
        grads = tape.gradient(loss, vars)

        if numpy.isnan(loss.numpy()):
            raise ValueError("Loss is NaN")
        self.optimizer.apply_gradients(zip(grads, vars))

        for key in info:
            tf.summary.scalar(key, info[key], step=self._step)
        self._step += 1
        return info

    def get_loss(self, in_, out):
        return self.config.target.loss(self.model, in_, out)

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

        if not self.model.scalers_fitted:
            argsin, argsout = next(iter(d.batch(10_000)))
            self.model.fit_scalers(argsin, argsout)

        for i in range(self.config.epochs):
            logging.info(f"Epoch {i+1}")
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
                if isinstance(self.model, UVTQSimple):
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
        self.logger.log_profiles(key, data, step)

    def log_dict(self, prefix, metrics, step):
        self.logger.log_dict(prefix, metrics, step)

    def partial_fit(self, statein: State, stateout: State):

        in_tensors = _xarray_to_tensor(statein, self.input_variables)
        out_tensors = _xarray_to_tensor(stateout, self.output_variables)
        d = tf.data.Dataset.from_tensor_slices((in_tensors, out_tensors)).shuffle(
            1_000_000
        )
        self.batch_fit(d)

    def predict(self, state: State) -> State:
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

    @property
    def _checkpoint(self) -> tf.train.Checkpoint:
        return tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    _config = "config.yaml"
    _model = "model"

    def dump(self, path: str):
        if path:
            os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, self._config), "w") as f:
            json.dump(dataclasses.asdict(self.config), f)
        self._checkpoint.write(os.path.join(path, self._model))

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, cls._config), "r") as f:
            config = OnlineEmulatorConfig.from_dict(json.load(f))
        model = cls(config)
        model._checkpoint.read(os.path.join(path, cls._model))
        return model


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
        self.scalers_fitted = False
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
        self.scalers_fitted = True

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


class ScalarMLP(tf.keras.layers.Layer):
    def __init__(self, num_hidden=256, num_hidden_layers=1, var_number=0, var_level=0):
        super(ScalarMLP, self).__init__()
        self.scalers_fitted = False
        self.sequential = tf.keras.Sequential()

        # output level
        self.var_number = var_number
        self.var_level = var_level

        # input and output normalizations
        self.norm = NormLayer(name="norm")
        self.output_scaler = ScalarNormLayer(name="output_scalar")

        # model architecture
        self.sequential.add(self.norm)

        for _ in range(num_hidden_layers):
            self.sequential.add(tf.keras.layers.Dense(num_hidden, activation="relu"))

        self.sequential.add(tf.keras.layers.Dense(1, name="out"))
        self.sequential.add(self.output_scaler)

    def call(self, args: Sequence[tf.Variable]):
        # assume has dims: batch, z
        args = [atleast_2d(arg) for arg in args]
        stacked = tf.concat(args, axis=-1)
        t0 = args[self.var_number][:, self.var_level : self.var_level + 1]
        return t0 + self.sequential(stacked)

    def _fit_input_scaler(self, args: Sequence[tf.Variable]):
        args = [atleast_2d(arg) for arg in args]
        stacked = tf.concat(args, axis=-1)
        self.norm.fit(stacked)

    def _fit_output_scaler(self, argsin: Sequence[tf.Variable], argsout: tf.Variable):
        t0 = argsin[self.var_number][:, self.var_level : self.var_level + 1]
        t1 = argsout[self.var_number][:, self.var_level : self.var_level + 1]
        self.output_scaler.fit(t1 - t0)

    def fit_scalers(self, argsin: Sequence[tf.Variable], argsout: tf.Variable):
        self._fit_input_scaler(argsin)
        self._fit_output_scaler(argsin, argsout)
        self.scalers_fitted = True


def needs_restart(state) -> bool:
    """Detect if error state is happening, in which case we should restart the
    model from a clean state
    """
    return False


def get_model(config: OnlineEmulatorConfig) -> tf.keras.Model:
    if isinstance(config.target, MultiVariableLoss):
        logging.info("Using ScalerMLP")
        n = config.levels
        model = UVTQSimple(n, n, n, n)
    elif isinstance(config.target, ScalarLoss):
        logging.info("Using ScalerMLP")
        model = ScalarMLP(
            var_number=config.target.variable,
            var_level=config.target.level,
            num_hidden=config.num_hidden,
            num_hidden_layers=config.num_hidden_layers,
        )
    else:
        raise NotImplementedError(f"{config}")
    return model
