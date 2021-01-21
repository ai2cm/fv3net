from typing import Iterable, Type, Optional, Union
import logging
import numpy as np
import os
import random
import tensorflow as tf
from . import _models
from .._shared import Estimator

logger = logging.getLogger(__file__)

__all__ = ["get_model"]


def get_model(
    model_type: str,
    sample_dim_name: str,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    **hyperparameters
) -> Estimator:
    """Initialize and return a Estimator instance.

    Args:
        model_type: the type of model to return
        input_variables: input variable names
        output_variables: output variable names
        **hyperparameters: other settings relevant to the model_type chosen

    Returns:
        model
    """
    return get_model_class(model_type)(  # type: ignore
        sample_dim_name, input_variables, output_variables, **hyperparameters
    )


def get_model_class(model_type: str) -> Type[Estimator]:
    """Returns a class implementing the Estimator interface corresponding to the model type.
    
    Args:
        model_type: the type of model

    Returns:
        model_class: a subclass of Estimator corresponding to the model type
    """
    return getattr(_models, model_type)


def get_optimizer(hyperparameters: dict = None):
    hyperparameters = hyperparameters or {}
    optimizer_config = hyperparameters.pop("optimizer", {})
    if optimizer_config:
        optimizer_class = getattr(
            tf.keras.optimizers, optimizer_config.get("name", "Adam")
        )
        optimizer_kwargs = optimizer_config.get("kwargs", {})
        optimizer = optimizer_class(**optimizer_kwargs)
    else:
        optimizer = None
    return optimizer


def get_regularizer(
    hyperparameters: dict = None,
) -> Optional[tf.keras.regularizers.Regularizer]:
    # Will be assumed to be a kernel regularizer when used in the model
    hyperparameters = hyperparameters or {}
    regularizer_config = hyperparameters.pop("regularizer", {})
    if regularizer_config:
        regularizer_class = getattr(
            tf.keras.regularizers, regularizer_config.get("name", "L2")
        )
        regularizer_kwargs = regularizer_config.get("kwargs", {})
        regularizer = regularizer_class(**regularizer_kwargs)
    else:
        regularizer = None
    return regularizer


def set_random_seed(seed: Union[float, int] = 0):
    # https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    tf.random.set_seed(seed + 3)