import dataclasses
from typing import Callable, List, Mapping

import tensorflow as tf
from fv3fit.emulation.layers.normalization2 import (
    CenterMethod,
    NormFactory,
    NormLayer,
    ScaleMethod,
)

from .._shared.config import OptimizerConfig


class NormalizedMSE:
    """
    Keras MSE that uses an emulation normalization class before
    scoring
    """

    def __init__(self, norm_layer: NormLayer):
        self._normalize = norm_layer.forward
        self._mse = tf.keras.losses.MeanSquaredError()

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return self._mse(self._normalize(y_true), self._normalize(y_pred))


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
    normalization: NormFactory = NormFactory(ScaleMethod.all, CenterMethod.per_feature)
    loss_variables: List[str] = dataclasses.field(default_factory=list)
    metric_variables: List[str] = dataclasses.field(default_factory=list)
    weights: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def build(self, output_samples: Mapping[str, tf.Tensor],) -> Callable:
        """
        Prepare the normalized losses for each variable by creating a
        fitted NormalizedMSE object and place them into the respective
        loss (+ weights) or metrics group

        Args:
            output_names: names of each output the model produces
            output_samples: sample tensors for each output to fit
                the normalizing layer

        """
        loss_funcs = {}
        for out_varname, sample in output_samples.items():
            norm_layer = self.normalization.build(sample)
            loss_funcs[out_varname] = NormalizedMSE(norm_layer)

        return _MultiVariableLoss(
            loss_funcs=loss_funcs,
            loss_variables=self.loss_variables,
            metric_variables=self.metric_variables,
            weights=self.weights,
        )


ScalarLossFunction = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


class _MultiVariableLoss:
    def __init__(
        self,
        loss_funcs: Mapping[str, ScalarLossFunction],
        loss_variables: List[str],
        metric_variables: List[str],
        weights: Mapping[str, float],
    ):
        self.loss_funcs = loss_funcs
        self.loss_variables = loss_variables
        self.metric_variables = metric_variables
        self.weights = weights

    def __call__(self, truth, prediction):

        # add transformed variables to truth, but not prediction (which should
        # already have them)
        y = prediction

        metrics = {}
        loss = 0.0
        for out_varname in truth:
            if out_varname in self.loss_variables + self.metric_variables:
                loss_value = self.loss_funcs[out_varname](
                    truth[out_varname], y[out_varname]
                )
                weight = self.weights.get(out_varname, 1.0)
                if out_varname in self.loss_variables:
                    loss += loss_value * weight
                # append "_loss" for backwards compatibility
                metrics[out_varname + "_loss"] = loss_value
        return loss, metrics
