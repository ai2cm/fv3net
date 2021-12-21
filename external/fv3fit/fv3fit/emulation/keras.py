import dataclasses
import logging
import numpy as np
import os
import tensorflow as tf
from typing import Callable, Optional, Mapping, List, Sequence, Union

from fv3fit.emulation.layers.normalization import NormalizeConfig, standard_deviation_all_features
import fv3fit.keras.adapters
from .scoring import score_multi_output, ScoringOutput
from .._shared.config import OptimizerConfig
from toolz import get

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


class NormalizedMSE(tf.keras.losses.MeanSquaredError):
    """
    Keras MSE that uses an emulation normalization class before
    scoring
    """

    def __init__(self, norm_cls_name, sample_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._normalize = NormalizeConfig(norm_cls_name, sample_data).initialize_layer()

    def call(self, y_true, y_pred):
        return super().call(self._normalize(y_true), self._normalize(y_pred))


@dataclasses.dataclass
class CustomLoss:
    """
    Use custom custom normalized MSE-based losses for specified
    variables

    Args:
        optimizer: configuration for the optimizer to
            compile with the model
        normalization: the normalization type (see normalization.py) to
            use for the MSE
        loss_variables: variable names to include in the MSE loss dict
        metric_variables: variable names to include in the metrics dict
        weights: custom scaling for the loss variables applied in the
            overall keras "loss" term
    """

    optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    normalization: str = "mean_std"
    loss_variables: List[str] = dataclasses.field(default_factory=list)
    metric_variables: List[str] = dataclasses.field(default_factory=list)
    weights: Mapping[str, float] = dataclasses.field(default_factory=dict)
    _fitted: bool = dataclasses.field(init=False, default=False)

    def prepare(self, output_samples: Mapping[str, tf.Tensor]):
        """
        Prepare the normalized losses for each variable by creating a
        fitted NormalizedMSE object and place them into the respective
        loss (+ weights) or metrics group

        Args:
            output_names: names of each output the model produces
            output_samples: sample tensors for each output to fit
                the normalizing layer
             
        """
        losses = {}
        metrics = {}
        weights = {}
        for out_varname, sample in output_samples.items():
            loss_func = NormalizedMSE(self.normalization, sample)

            if out_varname in self.loss_variables:
                losses[out_varname] = loss_func

                if out_varname in self.weights:
                    weights[out_varname] = self.weights[out_varname]
                else:
                    weights[out_varname] = 1.0

            elif out_varname in self.metric_variables:
                metrics[out_varname] = loss_func

        self._loss = losses
        self._metrics = metrics
        self._weights = weights
        self._fitted = True

    def compile(self, model: tf.keras.Model):
        if not self._fitted:
            raise ValueError(
                "Cannot compile custom loss without first calling prepare()."
            )

        model.compile(
            loss=self._loss,
            metrics=self._metrics,
            loss_weights=self._weights,
            optimizer=self.optimizer.instance,
        )


KerasMetrics = List[str]
KerasWeights = Union[Mapping[str, float], List[float]]


@dataclasses.dataclass
class StandardLoss:
    """Standard loss configuration provided to a tf.keras.Model.compile"""

    optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    loss: Optional[str] = None
    metrics: Optional[KerasMetrics] = None
    weights: Optional[KerasWeights] = None

    def prepare(self, **kwargs):
        """Nothing to do here"""
        pass

    def compile(self, model: tf.keras.Model):

        model.compile(
            loss=self.loss,
            metrics=self.metrics,
            loss_weights=self.weights,
            optimizer=self.optimizer.instance,
        )


ModelType = Callable[[Mapping[str, tf.Tensor]], Mapping[str, tf.Tensor]]
OutputSensitivity = Mapping[str, np.ndarray]


def get_jacobians(model: ModelType, inputs: Mapping[str, tf.Tensor]) -> Mapping[str, OutputSensitivity]:
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


def normalize_jacobians(
    all_jacobians: Mapping[str, OutputSensitivity],
    sample: Mapping[str, tf.Tensor],
) -> Mapping[str, OutputSensitivity]:
    """
    Generate sensitivity jacobions for each output of a model and
    normalize for easy inter-variable comparison.

    Normalization scaling uses the standard deviation across all
    de-meaned features for both the input (std_input) and output
    (std_output) sample, scaling the associated jacobian result
    by [ std_input / std_output ].
    """

    # normalize factors so sensitivities are comparable but still
    # preserve level-relative magnitudes
    normalize_factors = {
        name: float(standard_deviation_all_features(data))
        for name, data in sample.items()
    }

    normalized_jacobians = {}
    for out_name, per_input_jacobians in all_jacobians.items():
        for in_name, j in per_input_jacobians.items():
            # multiply d_output/d_input by std_input/std_output
            factor = normalize_factors[in_name] / normalize_factors[out_name]
            normalized_jacobians.setdefault(out_name, {})[in_name] = j * factor

    return normalized_jacobians
