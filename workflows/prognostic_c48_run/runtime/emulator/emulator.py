from collections import defaultdict
import dataclasses
from typing import (
    Callable,
    Hashable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
    List,
)
import os
from runtime.names import SPHUM
import xarray as xr
import numpy
import tensorflow as tf
import dacite
from runtime.emulator.batch import (
    get_prognostic_variables,
    batch_to_specific_humidity_basis,
    to_tensors,
    to_dict_no_static_vars,
)
from runtime.emulator.loggers import WandBLogger, ConsoleLogger, TBLogger, LoggerList
from runtime.emulator.loss import RHLoss, QVLoss, MultiVariableLoss
from runtime.emulator.models import UVTQSimple, UVTRHSimple, ScalarMLP, RHScalarMLP
from runtime.emulator.models import V1QCModel
import logging
import json


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
            QVLoss targets.
        wandb_logger: if True, then enable weights and biases saving
        checkpoint: path to model artifact in Weights and biases
            "<entity>/<project>/<name>:tag"

    """

    batch_size: int = 64
    learning_rate: float = 0.01
    num_hidden: int = 256
    num_hidden_layers: int = 1
    momentum: float = 0.5
    # online parameters
    online: bool = False
    train: bool = True

    # will ignore the emulator for any z larger than this value
    # remember higher z is lower in the atmosphere hence "below"
    ignore_humidity_below: Optional[int] = None

    # other parameters
    extra_input_variables: List[str] = dataclasses.field(default_factory=list)
    epochs: int = 1
    levels: int = 79
    batch: Optional[BatchDataConfig] = None
    target: Union[MultiVariableLoss, QVLoss, RHLoss] = dataclasses.field(
        default_factory=MultiVariableLoss
    )
    relative_humidity: bool = False
    wandb_logger: bool = False

    output_path: str = ""
    checkpoint: Optional[str] = None
    weight_sharing: bool = False
    cloud_water: bool = False

    @property
    def input_variables(self) -> List[str]:
        return get_prognostic_variables() + list(self.extra_input_variables)

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
        group.add_argument("--qc-weight", default=1e9, type=float)
        group.add_argument("--rh-weight", default=0.0, type=float)
        group.add_argument("--u-weight", default=100.0, type=float)
        group.add_argument("--v-weight", default=100.0, type=float)
        group.add_argument("--t-weight", default=100.0, type=float)
        group.add_argument("--levels", default="", type=str)
        group.add_argument(
            "--no-weight-sharing",
            dest="weight_sharing",
            action="store_true",
            help="Weights not shared by any inputs.",
        )

        group = parser.add_argument_group("single level output")
        group.add_argument(
            "--relative-humidity",
            action="store_true",
            help="if true use relative based prediction.",
        )
        group.add_argument(
            "--cloud-water",
            action="store_true",
            help="if true predict the cloud water field.",
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
        config.cloud_water = args.cloud_water

        if args.level:
            if args.relative_humidity:
                config.target = RHLoss(level=args.level, scale=args.scale)
            else:
                config.target = QVLoss(args.level, scale=args.scale)
        elif args.multi_output:
            levels = [int(s) for s in args.levels.split(",") if s]
            config.target = MultiVariableLoss(
                levels=levels,
                qc_weight=args.qc_weight,
                q_weight=args.q_weight,
                u_weight=args.u_weight,
                v_weight=args.v_weight,
                t_weight=args.t_weight,
                rh_weight=args.rh_weight,
            )
            config.weight_sharing = args.weight_sharing
        else:
            raise NotImplementedError(
                f"No problem type detected. "
                "Need to pass either --level or --multi-output"
            )

        if args.extra_variables:
            config.extra_input_variables = args.extra_variables.split(",")

        return config


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
        self.output_variables: Sequence[str] = get_prognostic_variables()
        self.extra_inputs = config.extra_input_variables
        self._step = 0
        self.logger = LoggerList([TBLogger(), ConsoleLogger(), WandBLogger()])

        self.logger = LoggerList([TBLogger(), ConsoleLogger()])

        if config.wandb_logger:
            self.logger.loggers.append(WandBLogger())

    @property
    def input_variables(self):
        return self.config.input_variables

    def batch_to_specific_humidity_basis(self, x):
        return batch_to_specific_humidity_basis(x, self.extra_inputs)

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
            in_ = self.batch_to_specific_humidity_basis(x)
            out = batch_to_specific_humidity_basis(y)
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
            in_ = self.batch_to_specific_humidity_basis(argsin)
            out = batch_to_specific_humidity_basis(argsout)
            self.model.fit_scalers(in_, out)

        for i in range(self.config.epochs):
            logging.info(f"Epoch {i+1}")
            train_loss = defaultdict(lambda: [])
            for x, y in d.batch(self.config.batch_size):
                in_ = self.batch_to_specific_humidity_basis(x)
                out = batch_to_specific_humidity_basis(y)
                info = self.step(in_, out)
                for key in info:
                    train_loss[key].append(info[key])

            loss_epoch_train = average(train_loss)
            self.log_dict("train_epoch", loss_epoch_train, step=i)

            if validation_data:
                loss_epoch_test = self.score(validation_data)
                self.log_dict("test_epoch", loss_epoch_test, step=i)
                x, y = next(iter(validation_data.batch(3).take(1)))
                in_ = self.batch_to_specific_humidity_basis(x)
                out = batch_to_specific_humidity_basis(y)
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
        in_tensors = to_tensors(in_)
        x = self.batch_to_specific_humidity_basis(in_tensors)
        out = self.model(x)

        tensors = to_dict_no_static_vars(out)

        dims = ["sample", "z"]
        attrs = {"units": "no one cares"}

        return xr.Dataset(
            {key: (dims, val, attrs) for key, val in tensors.items()},
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


def _xarray_to_tensor(state, keys):
    in_ = stack(state, keys)
    return to_tensors(in_)


def stack(state: State, keys) -> xr.Dataset:
    ds = xr.Dataset({key: state[key] for key in keys})
    sample_dims = ["y", "x"]
    return ds.stack(sample=sample_dims).transpose("sample", ...)


def needs_restart(state) -> bool:
    """Detect if error state is happening, in which case we should restart the
    model from a clean state
    """
    return False


def get_model(config: OnlineEmulatorConfig) -> tf.keras.Model:
    if config.cloud_water:
        logging.info("Using V1QCModel")
        return V1QCModel(config.levels)
    elif isinstance(config.target, MultiVariableLoss):
        logging.info("Using ScalerMLP")
        n = config.levels
        if config.relative_humidity:
            return UVTRHSimple(n, n, n, n)
        else:
            return UVTQSimple(n, n, n, n)
    elif isinstance(config.target, QVLoss):
        logging.info("Using ScalerMLP")
        model = ScalarMLP(
            var_level=config.target.level,
            num_hidden=config.num_hidden,
            num_hidden_layers=config.num_hidden_layers,
        )
    elif isinstance(config.target, RHLoss):
        logging.info("Using RHScaler")
        model = RHScalarMLP(
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


def _update_state_with_emulator(
    state: MutableMapping[Hashable, xr.DataArray],
    src: Mapping[Hashable, xr.DataArray],
    from_orig: Callable[[xr.DataArray], xr.DataArray],
) -> None:
    """
    Args:
        state: the mutable state object
        src: updates to put into state
        from_orig: a function returning a mask. Where this mask is True, the
            original state array will be used.

    """
    for key in src:
        arr = state[key]
        mask = from_orig(key, arr)
        state[key] = arr.where(mask, src[key].variable)


@dataclasses.dataclass
class from_orig:
    ignore_humidity_below: Optional[int] = None

    def __call__(self, name: str, arr: xr.DataArray) -> xr.DataArray:
        if name == SPHUM:
            if self.ignore_humidity_below is not None:
                return arr.z > self.ignore_humidity_below
            else:
                return False
        else:
            return True


def update_state_with_emulator(
    state: MutableMapping[Hashable, xr.DataArray],
    src: Mapping[Hashable, xr.DataArray],
    ignore_humidity_below: Optional[int] = None,
) -> None:
    return _update_state_with_emulator(state, src, from_orig(ignore_humidity_below))
