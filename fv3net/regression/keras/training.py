from typing import Sequence, Iterable
import xarray as xr
from ..shared import ModelTrainingConfig
import logging
from . import models
from tf import keras
from ..sklearn.wrapper import _pack
import numpy as np
import loaders

logger = logging.getLogger(__file__)


def _get_model(train_config):
    return getattr(models, train_config.model_type)(**train_config.hyperparameters)


def train_model(
    batched_data: Sequence[xr.Dataset], train_config: ModelTrainingConfig
) -> models.Model:
    """
    Args:
        batched_data: Sequence of training batch datasets
        train_config: model training configuration
    Returns:
        trained sklearn model wrapper object
    """
    model = _get_model(train_config)
    X = loaders.FunctionOutputSequence(
        batched_data, lambda ds: ds[train_config.input_variables]
    )
    y = loaders.FunctionOutputSequence(
        batched_data, lambda ds: ds[train_config.output_variables]
    )
    model.fit(X, y)
    return model
