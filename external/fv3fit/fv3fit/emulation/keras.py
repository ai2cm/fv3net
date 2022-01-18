import logging
import os
import tensorflow as tf
from toolz import get
from typing import Mapping

from .scoring import score_multi_output, ScoringOutput
import fv3fit.keras.adapters

logger = logging.getLogger(__name__)


def save_model(model: tf.keras.Model, destination: str):

    """
    Remove any compiled options and save model under "model.tf"
    to a destination for standardization.  Custom losses/metricss
    require custom object resolution during load, so it's better
    to remove.

    https://github.com/tensorflow/tensorflow/issues/43478

    Args:
        model: tensorflow model
        destination: path to store "model.tf" under
    """
    # clear all the weights and optimizers settings
    model.compile()
    model_path = os.path.join(destination, "model.tf")
    logging.getLogger(__name__).debug(f"saving model to {model_path}")
    model.save(model_path, save_format="tf")
    return model_path


def score_model(model: tf.keras.Model, data: Mapping[str, tf.Tensor],) -> ScoringOutput:
    """
    Score an emulation model with single or multiple
    output tensors.  Created to handle difference between
    single-out and multiple-out models producing a tensor
    vs. a list

    Args:
        model: tensorflow emulation model
        data: data to score with, must contain inputs and outputs of
        ``model``.
    """
    model = fv3fit.keras.adapters.ensure_dict_output(model)
    prediction = model(data)
    names = sorted(set(prediction) & set(data))
    return score_multi_output(get(names, data), get(names, prediction), names)
