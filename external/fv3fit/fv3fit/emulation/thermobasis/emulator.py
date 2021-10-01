from collections import defaultdict
import dataclasses
import json
import logging
from typing import (
    Optional,
    Sequence,
    Union,
    List,
)
import os
import numpy
import tensorflow as tf
import dacite
from fv3fit.emulation.thermobasis.batch import (
    get_prognostic_variables,
    batch_to_specific_humidity_basis,
)
from fv3fit.emulation.thermobasis.loggers import (
    WandBLogger,
    ConsoleLogger,
    TBLogger,
    LoggerList,
)
from fv3fit.emulation.thermobasis.loss import (
    RHLossSingleLevel,
    QVLossSingleLevel,
    MultiVariableLoss,
)
from fv3fit.emulation.thermobasis.models import (
    UVTQSimple,
    UVTRHSimple,
    ScalarMLP,
    RHScalarMLP,
)
from fv3fit.emulation.thermobasis.models import V1QCModel


def average(metrics):
    return {key: sum(metrics[key]) / len(metrics[key]) for key in metrics}


@dataclasses.dataclass
class BatchDataConfig:
    training_path: str
    testing_path: str


@dataclasses.dataclass
class Config:
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

    """

    batch_size: int = 64
    learning_rate: float = 0.01
    num_hidden: int = 256
    num_hidden_layers: int = 1
    momentum: float = 0.5

    # other parameters
    extra_input_variables: List[str] = dataclasses.field(default_factory=list)
    epochs: int = 1
    levels: int = 79
    batch: Optional[BatchDataConfig] = None
    target: Union[
        MultiVariableLoss, QVLossSingleLevel, RHLossSingleLevel
    ] = dataclasses.field(default_factory=MultiVariableLoss)
    relative_humidity: bool = False
    wandb_logger: bool = False

    output_path: str = ""
    weight_sharing: bool = False
    cloud_water: bool = False
    l2: Optional[float] = None

    @property
    def input_variables(self) -> List[str]:
        return get_prognostic_variables() + list(self.extra_input_variables)

    @classmethod
    def from_dict(cls, dict_) -> "Config":
        return dacite.from_dict(cls, dict_, dacite.Config(strict=True))

    @staticmethod
    def register_parser(parser):
        parser.add_argument(
            "--training-data",
            default="data/training",
            help="path to directory of netcdf files to train from.",
        )
        parser.add_argument(
            "--testing-data",
            default="data/validation",
            help="same as --testing-data but used for validation.",
        )
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

        # other options
        parser.add_argument(
            "--l2", default=None, type=float, help="l2 penalty for the weight matrices"
        )

    @staticmethod
    def from_args(args) -> "Config":
        config = Config()
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
        config.l2 = args.l2

        if args.level:
            if args.relative_humidity:
                config.target = RHLossSingleLevel(level=args.level, scale=args.scale)
            else:
                config.target = QVLossSingleLevel(args.level, scale=args.scale)
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


class Trainer:
    """A training loop that supports online and offline training of emulator
    models.

    To add a new emulation model (taking in and predicting ThermoBasis objects),
    add a section to ``get_model`` that returns a model with the same interface
    as e.g. :py:class:`V1QCModel`.
    """

    def __init__(
        self, config: Config,
    ):
        self.config = config
        self.model = get_model(config)
        self.optimizer = tf.optimizers.SGD(
            learning_rate=config.learning_rate, momentum=config.momentum
        )
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
        prediction = self.model(in_)
        prediction_loss, info = self.config.target.loss(prediction, out)
        return prediction_loss + sum(self.model.losses), info

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
            config = Config.from_dict(json.load(f))
        model = cls(config)
        model._checkpoint.read(os.path.join(path, cls._model))
        return model


def get_regularizer(config: Config) -> Optional[tf.keras.regularizers.Regularizer]:
    return tf.keras.regularizers.L2(config.l2) if config.l2 else None


def get_model(config: Config) -> tf.keras.Model:
    regularizer = get_regularizer(config)
    if config.cloud_water:
        logging.info("Using V1QCModel")
        return V1QCModel(config.levels, regularizer=regularizer)
    elif isinstance(config.target, MultiVariableLoss):
        logging.info("Using ScalerMLP")
        n = config.levels
        if config.relative_humidity:
            return UVTRHSimple(n, n, n, n, regularizer=regularizer)
        else:
            return UVTQSimple(n, n, n, n, regularizer=regularizer)
    elif isinstance(config.target, QVLossSingleLevel):
        logging.info("Using ScalerMLP")
        model = ScalarMLP(
            var_level=config.target.level,
            num_hidden=config.num_hidden,
            num_hidden_layers=config.num_hidden_layers,
        )
    elif isinstance(config.target, RHLossSingleLevel):
        logging.info("Using RHScaler")
        model = RHScalarMLP(
            var_level=config.target.level,
            num_hidden=config.num_hidden,
            num_hidden_layers=config.num_hidden_layers,
        )
    else:
        raise NotImplementedError(f"{config}")
    return model
