from typing import Iterable, Type
import logging
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
