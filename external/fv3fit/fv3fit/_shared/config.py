import dataclasses
from typing_extensions import Literal
import fsspec
import yaml
import os
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Tuple,
    Union,
    Sequence,
    List,
    Type,
    Dict,
)
from fv3fit.typing import Dataclass
import xarray as xr
from .predictor import Predictor
import dacite
import numpy as np
import random
import loaders

# TODO: move all keras configs under fv3fit.keras
import tensorflow as tf


DELP = "pressure_thickness_of_atmospheric_layer"
MODEL_CONFIG_FILENAME = "training_config.yml"


TrainingFunction = Callable[
    [
        Sequence[str],
        Sequence[str],
        Dataclass,
        Sequence[xr.Dataset],
        Sequence[xr.Dataset],
    ],
    Predictor,
]


def set_random_seed(seed: Union[float, int] = 0):
    # https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    tf.random.set_seed(seed + 3)


# TODO: delete this routine by refactoring the tests to no longer depend on it
def get_keras_model(name):
    return TRAINING_FUNCTIONS[name][0]


@dataclasses.dataclass
class TrainingConfig:
    """Convenience wrapper for model training parameters and file info

    Attrs:
        model_type: sklearn model type or keras model class to initialize
        input_variables: variables used as features
        output_variables: variables to predict
        hyperparameters: model_type-specific training configuration
        additional_variables: variables needed for training which aren't input
            or output variables of the trained model
        sample_dim_name: deprecated, internal name used for sample dimension
            when training and predicting
        random_seed: value to use to initialize randomness
    """

    model_type: str
    input_variables: List[str]
    output_variables: List[str]
    hyperparameters: Dataclass
    additional_variables: List[str] = dataclasses.field(default_factory=list)
    sample_dim_name: str = "sample"
    random_seed: Union[float, int] = 0

    @classmethod
    def from_dict(cls, kwargs) -> "TrainingConfig":
        kwargs = {**kwargs}  # make a copy to avoid mutating the input
        hyperparameter_class = get_hyperparameter_class(kwargs["model_type"])
        kwargs["hyperparameters"] = dacite.from_dict(
            data_class=hyperparameter_class, data=kwargs.get("hyperparameters", {})
        )
        return dacite.from_dict(data_class=cls, data=kwargs)


TRAINING_FUNCTIONS: Dict[str, Tuple[TrainingFunction, Type[Dataclass]]] = {}


def get_hyperparameter_class(model_type: str) -> Type:
    if model_type in TRAINING_FUNCTIONS:
        _, subclass = TRAINING_FUNCTIONS[model_type]
    else:
        raise ValueError(f"unknown model_type {model_type}")
    return subclass


def get_training_function(model_type: str) -> TrainingFunction:
    if model_type in TRAINING_FUNCTIONS:
        estimator_class, _ = TRAINING_FUNCTIONS[model_type]
    else:
        raise ValueError(f"unknown model_type {model_type}")
    return estimator_class


def register_training_function(name: str, hyperparameter_class: type):
    """
    Returns a decorator that will register the given training function
    to be usable in training configuration.
    """

    def decorator(func: TrainingFunction) -> TrainingFunction:
        TRAINING_FUNCTIONS[name] = (func, hyperparameter_class)
        return func

    return decorator


@dataclasses.dataclass
class OptimizerConfig:
    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def instance(self) -> tf.keras.optimizers.Optimizer:
        cls = getattr(tf.keras.optimizers, self.name)
        return cls(**self.kwargs)


@dataclasses.dataclass
class RegularizerConfig:
    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def instance(self) -> tf.keras.regularizers.Regularizer:
        cls = getattr(tf.keras.regularizers, self.name)
        return cls(**self.kwargs)


# TODO: move this class to where the Dense training is defined when config.py
# no longer depends on it (i.e. when _ModelTrainingConfig is deleted)
@dataclasses.dataclass
class DenseHyperparameters:
    """
    Configuration for training a dense neural network based model.

    Args:
        weights: loss function weights, defined as a dict whose keys are
            variable names and values are either a scalar referring to the total
            weight of the variable. Default is a total weight of 1
            for each variable.
        normalize_loss: if True (default), normalize outputs by their standard
            deviation before computing the loss function
        optimizer_config: selection of algorithm to be used in gradient descent
        kernel_regularizer_config: selection of regularizer for hidden dense layer
            weights, by default no regularization is applied
        depth: number of dense layers to use between the input and output layer.
            The number of hidden layers will be (depth - 1)
        width: number of neurons to use on layers between the input and output layer
        gaussian_noise: how much gaussian noise to add before each Dense layer,
            apart from the output layer
        loss: loss function to use, should be 'mse' or 'mae'
        spectral_normalization: whether to apply spectral normalization to hidden layers
        save_model_checkpoints: if True, save one model per epoch when
            dumping, under a 'model_checkpoints' subdirectory
        nonnegative_outputs: if True, add a ReLU activation layer as the last layer
            after output denormalization layer to ensure outputs are always >=0
            Defaults to False.
        fit_kwargs: other keyword arguments to be passed to the underlying
            tf.keras.Model.fit() method
    """

    weights: Optional[Mapping[str, Union[int, float]]] = None
    normalize_loss: bool = True
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    kernel_regularizer_config: Optional[RegularizerConfig] = None
    depth: int = 3
    width: int = 16
    epochs: int = 3
    gaussian_noise: float = 0.0
    loss: str = "mse"
    spectral_normalization: bool = False
    save_model_checkpoints: bool = False
    nonnegative_outputs: bool = False

    # TODO: remove fit_kwargs by fixing how validation data is passed
    fit_kwargs: Optional[dict] = None


@dataclasses.dataclass
class RandomForestHyperparameters:
    """
    Configuration for training a random forest based model.

    Trains one random forest for each training batch provided.

    For more information about these settings, see documentation for
    `sklearn.ensemble.RandomForestRegressor`.

    Args:
        scaler_type: scaler to use for training
        scaler_kwargs: keyword arguments to pass to scaler initialization
        n_jobs: number of jobs to run in parallel when training a single random forest
        random_state: random seed to use when building trees, will be
            deterministically perturbed for each training batch
        n_estimators: the number of trees in each forest
        max_depth: maximum depth of each tree, by default is unlimited
        min_samples_split: minimum number of samples required to split an internal node
        min_samples_leaf: minimum number of samples required to be at a leaf node
        max_features: number of features to consider when looking for the best split
        max_samples: if bootstrap is True, number of samples to draw
            for each base estimator
        bootstrap: whether bootstrap samples are used when building trees.
            If False, the whole dataset is used to build each tree.
    """

    scaler_type: str = "standard"
    scaler_kwargs: Optional[Mapping] = None

    # don't set default to -1 because it causes non-reproducible training
    n_jobs: int = 8
    random_state: int = 0
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    max_features: Union[Literal["auto", "sqrt", "log2"], int, float] = "auto"
    max_samples: Optional[Union[int, float]] = None
    bootstrap: bool = True


@dataclasses.dataclass
class _ModelTrainingConfig:
    """Convenience wrapper for model training parameters and file info

    Attrs:
        model_type: sklearn model type or keras model class to initialize
        hyperparameters: arguments to pass to model class at initialization
            time
        input_variables: variables used as features
        output_variables: variables to predict
        batch_function: name of function from `fv3fit.batches` to use for
            loading batched data
        batch_kwargs: keyword arguments to pass to batch function
        data_path: location of training data to be loaded by batch function
        scaler_type: scaler to use for training
        scaler_kwargs: keyword arguments to pass to scaler initialization
        additional_variables: list of needed variables which are not inputs
            or outputs (e.g. pressure thickness if needed for scaling)
        random_seed: value to use to initialize randomness
        validation_timesteps: timestamps to use as validation samples
        save_model_checkpoints: whether to save a copy of the model at
            each epoch
        model_path: output location for final model
        timesteps_source: one of "timesteps_file",
            "sampled_outside_input_config", "input_config", "all_mapper_times"
    """

    model_type: str
    hyperparameters: dict
    input_variables: List[str]
    output_variables: List[str]
    batch_function: str
    batch_kwargs: dict
    data_path: Optional[str] = None
    scaler_type: str = "standard"
    scaler_kwargs: dict = dataclasses.field(default_factory=dict)
    additional_variables: List[str] = dataclasses.field(default_factory=list)
    random_seed: Union[float, int] = 0
    validation_timesteps: Sequence[str] = dataclasses.field(default_factory=list)
    save_model_checkpoints: bool = False
    model_path: str = ""
    timesteps_source: str = "timesteps_file"

    def __post_init__(self):
        if self.scaler_type == "mass":
            if DELP not in self.additional_variables:
                self.additional_variables.append(DELP)

    def asdict(self):
        return dataclasses.asdict(self)

    def dump(self, path: str, filename: str = None) -> None:
        dict_ = self.asdict()
        if filename is None:
            filename = MODEL_CONFIG_FILENAME
        with fsspec.open(os.path.join(path, filename), "w") as f:
            yaml.safe_dump(dict_, f)

    @classmethod
    def load(cls, path: str) -> "_ModelTrainingConfig":
        with fsspec.open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return _ModelTrainingConfig(**config_dict)


def legacy_config_to_new_config(legacy_config: _ModelTrainingConfig) -> TrainingConfig:
    config_class = TRAINING_FUNCTIONS[legacy_config.model_type][1]
    keys = [
        "model_type",
        "hyperparameters",
        "input_variables",
        "output_variables",
        "additional_variables",
        "random_seed",
        "sample_dim_name",
    ]
    if config_class is RandomForestHyperparameters:
        for key in ("scaler_type", "scaler_kwargs"):
            legacy_config.hyperparameters[key] = getattr(legacy_config, key)
    elif config_class is DenseHyperparameters:
        legacy_config.hyperparameters[
            "save_model_checkpoints"
        ] = legacy_config.save_model_checkpoints
        fit_kwargs = legacy_config.hyperparameters.pop("fit_kwargs", {})
        if legacy_config.validation_timesteps is not None:
            fit_kwargs["validation_dataset"] = validation_dataset(legacy_config)
        legacy_config.hyperparameters["fit_kwargs"] = fit_kwargs
    else:
        raise NotImplementedError(f"unknown model type {legacy_config.model_type}")
    config_dict = dataclasses.asdict(legacy_config)
    config_dict["sample_dim_name"] = "sample"
    training_config = TrainingConfig.from_dict(
        {key: config_dict[key] for key in keys if key in config_dict}
    )
    return training_config


def load_configs(
    config_path: str,
    data_path: str,
    output_data_path: str,
    timesteps_file=None,
    validation_timesteps_file=None,
) -> Tuple[TrainingConfig, loaders.BatchesConfig, Optional[loaders.BatchesConfig]]:
    """Load training configuration information from a legacy yaml config path.

    Dumps the legacy configuration class to the output_data_path.
    """
    # TODO: remove output_data_path argument, we need it here at the moment
    # to dump legacy_config before it gets a Dataset attached to it,
    # for backwards compatibility
    # we shouldn't need this when validation_dataset is in its own data config
    # and not attached to fit_kwargs
    legacy_config = _ModelTrainingConfig.load(config_path)
    legacy_config.data_path = data_path
    legacy_config.dump(output_data_path)
    config_dict = dataclasses.asdict(legacy_config)
    training_config = legacy_config_to_new_config(legacy_config)

    data_path = config_dict["data_path"]
    batches_function = config_dict["batch_function"]
    batches_kwargs = config_dict["batch_kwargs"]

    train_batches_kwargs = {**batches_kwargs}
    if timesteps_file is not None:
        with open(timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        train_batches_kwargs["timesteps"] = timesteps
    train_data_config = loaders.BatchesConfig(
        data_path=data_path,
        batches_function=batches_function,
        batches_kwargs=train_batches_kwargs,
    )

    if validation_timesteps_file is not None:
        validation_batches_kwargs = {**batches_kwargs}
        with open(validation_timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        validation_batches_kwargs["timesteps"] = timesteps
        validation_data_config: Optional[loaders.BatchesConfig] = loaders.BatchesConfig(
            data_path=data_path,
            batches_function=batches_function,
            batches_kwargs=validation_batches_kwargs,
        )
    else:
        validation_data_config = None

    return training_config, train_data_config, validation_data_config


# TODO: this should be made to work regardless of whether we're using
# keras or sklearn models, find a way to delete this code entirely and
# use the validation BatchesConfig instead.


def check_validation_train_overlap(
    train: Sequence[str], validate: Sequence[str]
) -> None:
    overlap = set(train) & set(validate)
    if overlap:
        raise ValueError(
            f"Timestep(s) {overlap} are in both train and validation sets."
        )


def validation_timesteps_config(train_config):
    val_config = legacy_config_to_batches_config(train_config)
    assert not isinstance(val_config.data_path, list)
    val_config.batches_kwargs["timesteps"] = train_config.validation_timesteps
    val_config.batches_kwargs["timesteps_per_batch"] = len(
        train_config.validation_timesteps  # type: ignore
    )
    return val_config


# TODO: refactor all tests and code using this to create BatchesConfig
# from the beginning and delete this helper routine
def legacy_config_to_batches_config(
    legacy_config: _ModelTrainingConfig,
) -> loaders.BatchesConfig:
    return loaders.BatchesConfig(
        data_path=str(legacy_config.data_path),
        batches_function=legacy_config.batch_function,
        batches_kwargs=legacy_config.batch_kwargs,
    )


def validation_dataset(train_config: _ModelTrainingConfig,) -> Optional[xr.Dataset]:
    if len(train_config.validation_timesteps) > 0:
        check_validation_train_overlap(
            train_config.batch_kwargs["timesteps"], train_config.validation_timesteps,
        )
        validation_config = validation_timesteps_config(train_config)
        # validation config puts all data in one batch
        validation_dataset_sequence = validation_config.load_batches(
            variables=train_config.input_variables
            + train_config.output_variables
            + train_config.additional_variables
        )
        if len(validation_dataset_sequence) > 1:
            raise ValueError(
                "Something went wrong! "
                "All validation data should be concatenated into a single batch. "
                f"There are {len(validation_dataset_sequence)} "
                "batches in the sequence."
            )
        return validation_dataset_sequence[0]
    else:
        validation_dataset = None
    return validation_dataset


def load_training_config(model_path: str) -> _ModelTrainingConfig:
    """Load training configuration information from a model directory URL.
    Note:
        This loads a file that you would get from using ModelTrainingConfig.dump
        with no filename argument, as is done by fv3fit.train. To ensure
        backwards compatibility, you should use this routine to load such
        a file instead of manually specifying the filename.
        The default filename may change in the future.
    Args:
        model_path: model dir dumped by fv3fit.dump
    Returns:
        dict: training config dict
    """
    config_path = os.path.join(model_path, MODEL_CONFIG_FILENAME)
    return _ModelTrainingConfig.load(config_path)
