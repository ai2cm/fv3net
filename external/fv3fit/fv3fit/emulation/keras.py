import dataclasses
import logging
import os
import tensorflow as tf
from typing import Mapping, List

from fv3fit.emulation.layers.normalization import NormalizeConfig
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
        self.loss_funcs = {}
        for out_varname, sample in output_samples.items():
            self.loss_funcs[out_varname] = NormalizedMSE(self.normalization, sample)

    def __call__(self, x, y):
        try:
            self.loss_funcs
        except AttributeError:
            raise ValueError("Cannot compute loss without first calling prepare().")

        metrics = {}
        loss = 0.0
        for out_varname in x:
            if out_varname in self.loss_variables + self.metric_variables:
                loss_value = self.loss_funcs[out_varname](
                    x[out_varname], y[out_varname]
                )
                weight = self.weights.get(out_varname, 1.0)
                if out_varname in self.loss_variables:
                    loss += loss_value * weight
                metrics[out_varname] = loss_value
        return loss, metrics
