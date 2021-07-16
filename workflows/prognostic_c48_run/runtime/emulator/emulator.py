from collections import defaultdict
import dataclasses
from typing import Mapping, Optional, Sequence, Tuple, Union, List
import os
import xarray as xr
import numpy
import tensorflow as tf
import dacite
from runtime.emulator.loggers import WandBLogger, ConsoleLogger, TBLogger, LoggerList
from runtime.emulator.loss import RHLoss, ScalarLoss, MultiVariableLoss
from runtime.emulator.thermo import (
    RelativeHumidityBasis,
    SpecificHumidityBasis,
    ThermoBasis,
)
import logging
import json


U = "eastward_wind"
V = "northward_wind"
T = "air_temperature"
Q = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
DELZ = "vertical_thickness_of_atmospheric_layer"


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
        train: if True the model is trained online
        batch: if provided then these data are used for training the ML model
        num_hidden_layers: number of hidden layers used. Only implemented for
            ScalarLoss targets.
        wandb_logger: if True, then enable weights and biases saving
        checkpoint: path to model artifact in Weights and biases
            "<entity>/<project>/<name>:tag"

    """

    batch_size: int = 64
    learning_rate: float = 0.01
    num_hidden: int = 256
    num_hidden_layers: int = 1
    momentum: float = 0.5
    online: bool = False
    train: bool = True
    extra_input_variables: List[str] = dataclasses.field(default_factory=list)
    epochs: int = 1
    levels: int = 79
    batch: Optional[BatchDataConfig] = None
    target: Union[MultiVariableLoss, ScalarLoss, RHLoss] = dataclasses.field(
        default_factory=MultiVariableLoss
    )
    relative_humidity: bool = False
    wandb_logger: bool = False

    output_path: str = ""
    checkpoint: Optional[str] = None

    @property
    def input_variables(self) -> List[str]:
        return [U, V, T, Q, DELP, DELZ] + list(self.extra_input_variables)

    @classmethod
    def from_dict(cls, dict_) -> "OnlineEmulatorConfig":
        return dacite.from_dict(cls, dict_, dacite.Config(strict=True))

    @staticmethod
    def register_parser(parser):
        parser.add_argument("--training-data", default="data/training")
        parser.add_argument("--testing-data", default="data/validation")
        parser.add_argument("--batch-size", default=32, type=int)
        parser.add_argument("--epochs", default=60, type=int)
        parser.add_argument("--lr", default=0.01, type=float)
        parser.add_argument("--momentum", default=0.5, type=float)
        parser.add_argument("--timestep", default=900, type=int)
        parser.add_argument("--nfiles", default=0, type=int)
        parser.add_argument(
            "--wandb", action="store_true", help="Run with weights and biases logging."
        )

        group = parser.add_mutually_exclusive_group()
        group.add_argument("--multi-output", action="store_true")
        group.add_argument("--level", default=0, type=int, help="target level")

        group = parser.add_argument_group("multiple output target")
        group.add_argument("--q-weight", default=1e6, type=float)
        group.add_argument("--rh-weight", default=0.0, type=float)
        group.add_argument("--u-weight", default=100.0, type=float)
        group.add_argument("--v-weight", default=100.0, type=float)
        group.add_argument("--t-weight", default=100.0, type=float)
        group.add_argument("--levels", default="", type=str)

        group = parser.add_argument_group("single level output")
        group.add_argument(
            "--relative-humidity",
            action="store_true",
            help="if true use relative based prediction.",
        )
        group.add_argument("--variable", default=3, type=int)
        group.add_argument("--scale", default=1.0, type=float)

        group = parser.add_argument_group("network structure")
        group.add_argument("--num-hidden", default=256, type=int)
        group.add_argument("--num-hidden-layers", default=3, type=int)
        group.add_argument(
            "--extra-variables",
            default="",
            type=str,
            help="comma separated list of variable names.",
        )

    @staticmethod
    def from_args(args) -> "OnlineEmulatorConfig":
        config = OnlineEmulatorConfig()
        config.batch_size = args.batch_size
        config.epochs = args.epochs
        config.learning_rate = args.lr
        config.momentum = args.momentum
        config.batch = BatchDataConfig(args.training_data, args.testing_data)

        config.num_hidden = args.num_hidden
        config.num_hidden_layers = args.num_hidden_layers
        config.wandb_logger = args.wandb
        config.relative_humidity = args.relative_humidity

        if args.level:
            if args.relative_humidity:
                config.target = RHLoss(level=args.level, scale=args.scale)
            else:
                config.target = ScalarLoss(args.variable, args.level, scale=args.scale)
        elif args.multi_output:
            levels = [int(s) for s in args.levels.split(",") if s]
            config.target = MultiVariableLoss(
                levels=levels,
                q_weight=args.q_weight,
                u_weight=args.u_weight,
                v_weight=args.v_weight,
                t_weight=args.t_weight,
                rh_weight=args.rh_weight,
            )
        else:
            raise NotImplementedError(
                f"No problem type detected. "
                "Need to pass either --level or --multi-output"
            )

        if args.extra_variables:
            config.extra_input_variables = args.extra_variables.split(",")

        return config


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
        self.output_variables: Sequence[str] = (U, V, T, Q, DELP, DELZ)
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
        return self.config.target.loss(self.model(in_), out)

    def score(self, d: tf.data.Dataset):
        losses = defaultdict(list)
        for x, y in d.batch(10_000):
            in_ = SpecificHumidityBasis(x)
            out = SpecificHumidityBasis(y)
            _, info = self.get_loss(in_, out)
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
            # calls .build on any layers
            # self.model(argsin)
            in_ = SpecificHumidityBasis(argsin)
            out = SpecificHumidityBasis(argsout)
            self.model.fit_scalers(in_, out)

        for i in range(self.config.epochs):
            logging.info(f"Epoch {i+1}")
            train_loss = defaultdict(lambda: [])
            for x, y in d.batch(self.config.batch_size):
                in_ = SpecificHumidityBasis(x)
                out = SpecificHumidityBasis(y)
                info = self.step(in_, out)
                for key in info:
                    train_loss[key].append(info[key])

            loss_epoch_train = average(train_loss)
            self.log_dict("train_epoch", loss_epoch_train, step=i)

            if validation_data:
                loss_epoch_test = self.score(validation_data)
                self.log_dict("test_epoch", loss_epoch_test, step=i)
                x, y = next(iter(validation_data.batch(3).take(1)))
                in_ = SpecificHumidityBasis(x)
                out = SpecificHumidityBasis(y)
                pred = self.model(in_)
                if isinstance(self.model, UVTQSimple):
                    self.log_profiles(
                        "eastward_wind_truth", (out.u - in_.u).numpy().T, step=i
                    )
                    self.log_profiles(
                        "eastward_wind_prediction", (pred.u - in_.u).numpy().T, step=i
                    )
                    self.log_profiles(
                        "humidity_truth", (out.q - in_.q).numpy().T, step=i
                    )
                    self.log_profiles(
                        "humidity_prediction", (pred.q - in_.q).numpy().T, step=i
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
        in_tensors = to_tensors(in_, self.input_variables)
        x = SpecificHumidityBasis(in_tensors)
        out = self.model(x)
        dims = ["sample", "z"]

        attrs = {"units": "no one cares"}

        return xr.Dataset(
            {
                U: (dims, out.u, attrs),
                V: (dims, out.v, attrs),
                T: (dims, out.T, attrs),
                Q: (dims, out.q, attrs),
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

    def fit_scalers(self, x: ThermoBasis, y: ThermoBasis):
        self._fit_input_scaler(x.args)
        self._fit_output_scaler(x.args, y.args)
        self.scalers_fitted = True

    def call(self, in_: ThermoBasis) -> ThermoBasis:
        # assume has dims: batch, z
        args = [atleast_2d(arg) for arg in in_.args]
        stacked = tf.concat(args, axis=-1)
        hidden = self.relu(self.linear(self.norm(stacked)))

        return SpecificHumidityBasis(
            [
                in_.u + self.scalers[0](self.out_u(hidden)),
                in_.v + self.scalers[1](self.out_v(hidden)),
                in_.T + self.scalers[2](self.out_t(hidden)),
                in_.q + self.scalers[3](self.out_q(hidden)),
                in_.dp,
                in_.dz,
            ]
        )


class UVTRHSimple(UVTQSimple):
    def fit_scalers(self, x: ThermoBasis, y: ThermoBasis):
        self._fit_input_scaler(x.to_rh().args)
        self._fit_output_scaler(x.to_rh().args, y.to_rh().args)
        self.scalers_fitted = True

    def call(self, in_: ThermoBasis) -> ThermoBasis:
        # assume has dims: batch, z
        args = [atleast_2d(arg) for arg in in_.to_rh().args]
        stacked = tf.concat(args, axis=-1)
        hidden = self.relu(self.linear(self.norm(stacked)))

        return RelativeHumidityBasis(
            [
                in_.u + self.scalers[0](self.out_u(hidden)),
                in_.v + self.scalers[1](self.out_v(hidden)),
                in_.T + self.scalers[2](self.out_t(hidden)),
                in_.rh + self.scalers[3](self.out_q(hidden)),
                in_.rho,
                in_.dz,
            ]
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

    def call(self, in_: ThermoBasis):
        # assume has dims: batch, z
        args = [atleast_2d(arg) for arg in in_.args]
        stacked = tf.concat(args, axis=-1)
        t0 = in_.args[self.var_number][:, self.var_level : self.var_level + 1]
        return t0 + self.sequential(stacked)

    def _fit_input_scaler(self, in_: ThermoBasis):
        args = [atleast_2d(arg) for arg in in_.args]
        stacked = tf.concat(args, axis=-1)
        self.norm.fit(stacked)

    def _fit_output_scaler(self, argsin: ThermoBasis, argsout: ThermoBasis):
        t0 = argsin.args[self.var_number][:, self.var_level : self.var_level + 1]
        t1 = argsout.args[self.var_number][:, self.var_level : self.var_level + 1]
        self.output_scaler.fit(t1 - t0)

    def fit_scalers(self, argsin: ThermoBasis, argsout: ThermoBasis):
        self._fit_input_scaler(argsin)
        self._fit_output_scaler(argsin, argsout)
        self.scalers_fitted = True


class RHScalarMLP(ScalarMLP):
    def fit_scalers(self, argsin: ThermoBasis, argsout: ThermoBasis):
        rh_argsin = argsin.to_rh()
        rh_argsout = argsout.to_rh()
        super(RHScalarMLP, self).fit_scalers(rh_argsin, rh_argsout)

    def call(self, args: ThermoBasis):
        rh_args = args.to_rh()
        rh = super().call(rh_args)
        return rh


def needs_restart(state) -> bool:
    """Detect if error state is happening, in which case we should restart the
    model from a clean state
    """
    return False


def get_model(config: OnlineEmulatorConfig) -> tf.keras.Model:
    if isinstance(config.target, MultiVariableLoss):
        logging.info("Using ScalerMLP")
        n = config.levels
        if config.relative_humidity:
            return UVTRHSimple(n, n, n, n)
        else:
            return UVTQSimple(n, n, n, n)
    elif isinstance(config.target, ScalarLoss):
        logging.info("Using ScalerMLP")
        model = ScalarMLP(
            var_number=config.target.variable,
            var_level=config.target.level,
            num_hidden=config.num_hidden,
            num_hidden_layers=config.num_hidden_layers,
        )
    elif isinstance(config.target, RHLoss):
        logging.info("Using RHScaler")
        model = RHScalarMLP(
            var_number=3,
            var_level=config.target.level,
            num_hidden=config.num_hidden,
            num_hidden_layers=config.num_hidden_layers,
        )
    else:
        raise NotImplementedError(f"{config}")
    return model


def get_emulator(config: OnlineEmulatorConfig):
    if config.checkpoint:
        logging.info(f"Loading emulator from checkpoint {config.checkpoint}")
        return OnlineEmulator.load(config.checkpoint)
    else:
        return OnlineEmulator(config)
