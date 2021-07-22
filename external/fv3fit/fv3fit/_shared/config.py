import dataclasses
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

# TODO: move all keras configs under fv3fit.keras
import tensorflow as tf


DELP = "pressure_thickness_of_atmospheric_layer"


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
        derived_output_variables: optional list of prediction variables that
            are not directly predicted by the ML model but instead are derived
            using the ML-predicted output_variables
    """

    model_type: str
    input_variables: List[str]
    output_variables: List[str]
    hyperparameters: Dataclass
    additional_variables: List[str] = dataclasses.field(default_factory=list)
    sample_dim_name: str = "sample"
    random_seed: Union[float, int] = 0
    derived_output_variables: List[str] = dataclasses.field(default_factory=list)

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
        max_features: number of features to consider when looking for the best split,
            if string should be "sqrt", "log2", or "auto" (default), which correspond
            to the square root, log base 2, or total number of features respectively
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
    max_features: Union[str, int, float] = "auto"
    max_samples: Optional[Union[int, float]] = None
    bootstrap: bool = True
