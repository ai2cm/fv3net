from typing import Iterable, Type
import logging
from . import _models
from ._models.models import Model

logger = logging.getLogger(__file__)

__all__ = ["get_model", "get_model_class"]


def get_model(
    model_type: str,
    sample_dim_name: str,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    **hyperparameters
) -> Model:
    """Initialize and return a Model instance.

    Args:
        model_type: the type of model to return
        input_variables: input variable names
        output_variables: output variable names
        **hyperparameters: other settings relevant to the model_type chosen

    Returns:
        model
    """
    return get_model_class(model_type)(
        sample_dim_name, input_variables, output_variables, **hyperparameters
    )


def get_model_class(model_type: str) -> Type[Model]:
    """Returns a class implementing the Model interface corresponding to the model type.
    
    Args:
        model_type: the type of model

    Returns:
        model_class: a subclass of Model corresponding to the model type
    """
    return getattr(_models, model_type)
