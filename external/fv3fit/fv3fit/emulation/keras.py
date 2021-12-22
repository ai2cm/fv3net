import logging
import numpy as np
import os
import tensorflow as tf
from toolz import get
from typing import Callable, Dict, Mapping


from .layers.normalization import standard_deviation_all_features
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


ModelType = Callable[[Mapping[str, tf.Tensor]], Mapping[str, tf.Tensor]]
OutputSensitivity = Dict[str, np.ndarray]


def get_jacobians(
    model: ModelType, inputs: Mapping[str, tf.Tensor]
) -> Mapping[str, OutputSensitivity]:
    """
    Calculate jacobians for each output field relative to each
    model input:

    Args:
        model: model to calculate sensitivity matrices with
        inputs: inputs to calculate sensitivity against, expects
            tensors with dimensions of [1, nfeatures]
    """

    with tf.GradientTape(persistent=True) as g:
        g.watch(inputs)
        outputs = model(inputs)

    all_jacobians = {}
    for out_name, out_data in outputs.items():
        jacobians = g.jacobian(out_data, inputs)
        jacobians = {name: j[0, :, 0].numpy() for name, j in jacobians.items()}
        all_jacobians[out_name] = jacobians

    return all_jacobians


def standardize_jacobians(
    all_jacobians: Mapping[str, OutputSensitivity], sample: Mapping[str, tf.Tensor],
) -> Mapping[str, OutputSensitivity]:
    """
    Generate sensitivity jacobions for each output of a model and
    standardize (dimensionless) for easy inter-variable comparison.

    The scaling uses the standard deviation across all
    de-meaned features for both the input (std_input) and output
    (std_output) sample, scaling the associated jacobian result
    by [ std_input / std_output ].
    """

    # normalize factors so sensitivities are comparable but still
    # preserve level-relative magnitudes
    std_factors = {
        name: float(standard_deviation_all_features(data))
        for name, data in sample.items()
    }

    standardized_jacobians: Dict[str, OutputSensitivity] = {}
    for out_name, per_input_jacobians in all_jacobians.items():
        for in_name, j in per_input_jacobians.items():
            # multiply d_output/d_input by std_input/std_output
            factor = std_factors[in_name] / std_factors[out_name]
            standardized_jacobians.setdefault(out_name, {})[in_name] = j * factor

    return standardized_jacobians
