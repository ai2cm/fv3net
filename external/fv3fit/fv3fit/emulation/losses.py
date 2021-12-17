import dataclasses
from typing import List, Mapping

import tensorflow as tf
from fv3fit.emulation.layers.normalization import NormalizeConfig

from .._shared.config import OptimizerConfig


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
                # append "_loss" for backwards compatibility
                metrics[out_varname + "_loss"] = loss_value
        return loss, metrics
