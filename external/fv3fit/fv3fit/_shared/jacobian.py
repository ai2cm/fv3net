import numpy as np
import tensorflow as tf
from typing import Callable, Mapping, Dict

from ..emulation.layers.normalization import standard_deviation_all_features


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


def compute_standardized_jacobians(model, data, input_variables):
    avg_profiles = {
        name: tf.reduce_mean(data[name], axis=0, keepdims=True)
        for name in input_variables
    }
    jacobians = get_jacobians(model, avg_profiles)
    std_jacobians = standardize_jacobians(jacobians, data)
    return std_jacobians


def standardize_jacobians_by_vertical_level(
    all_jacobians: Mapping[str, OutputSensitivity],
    sample: Mapping[str, tf.Tensor],
    units: str = "dimensionless",
) -> Mapping[str, OutputSensitivity]:
    """
    Generate sensitivity jacobians for each output of a model and
    standardize (dimensionless or output units) for easy
    inter-variable comparison.

    The scaling uses the standard deviation of each vertical level
    for the input (std_input) and output
    (std_output) sample. The associated
    jacobian result is multiplied by [ std_input / std_output ].
    If units are "output" the associated jacobian result is multiplied
    by [ std_input ].
    """

    # normalize factors so sensitivities are comparable but still
    # preserve level-relative magnitudes
    std_factors = {
        name: tf.math.reduce_std(data, axis=0).numpy() for name, data in sample.items()
    }

    standardized_jacobians: Dict[str, OutputSensitivity] = {}
    for out_name, per_input_jacobians in all_jacobians.items():
        for in_name, j in per_input_jacobians.items():
            if units == "dimensionless":
                # multiply d_output/d_input at each level by std_input/std_output
                j_ = np.multiply(j, std_factors[in_name].reshape(1, -1))
                j_ = np.multiply(j_, 1.0 / std_factors[out_name].reshape(-1, 1))

            elif units == "output":
                # multiply d_output/d_input at each level by std_input
                j_ = np.multiply(j, std_factors[in_name].reshape(1, -1))
            else:
                raise ValueError(
                    "'units' arg must be either 'dimensionless' or 'output'"
                )
            standardized_jacobians.setdefault(out_name, {})[in_name] = j_

    return standardized_jacobians


def compute_vertically_standardized_jacobians(
    model, data, input_variables, units: str = "dimensionless"
):
    avg_profiles = {
        name: tf.reduce_mean(data[name], axis=0, keepdims=True)
        for name in input_variables
    }
    jacobians = get_jacobians(model, avg_profiles)
    std_jacobians = standardize_jacobians_by_vertical_level(jacobians, data, units)
    return std_jacobians
