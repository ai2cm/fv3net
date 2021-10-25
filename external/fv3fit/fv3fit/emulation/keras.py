import abc
import dataclasses
import logging
import os
import tensorflow as tf
from typing import Optional, Mapping, List, Sequence, Tuple, Union

from fv3fit.emulation.layers.normalization import NormalizeConfig
from .scoring import score_multi_output, score_single_output
from .._shared.config import OptimizerConfig

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

    model.compiled_loss = None
    model.compiled_metrics = None
    model.optimizer = None
    model_path = os.path.join(destination, "model.tf")
    model.save(model_path, save_format="tf")

    return model_path


def score_model(
    model: tf.keras.Model,
    inputs: Union[tf.Tensor, Tuple[tf.Tensor]],
    targets: Union[tf.Tensor, Sequence[tf.Tensor]],
):
    """
    Score an emulation model with single or multiple
    output tensors.  Created to handle difference between
    single-out and multiple-out models producing a tensor
    vs. a list

    Args:
        model: tensorflow emulation model
        inputs: model inputs
        targets: corresponding target tensor(s) for inputs 
    """

    prediction = model.predict(inputs)

    if len(model.output_names) > 1:
        scores, profiles = score_multi_output(targets, prediction, model.output_names)
    elif len(model.output_names) == 1:
        scores, profiles = score_single_output(
            targets, prediction, model.output_names[0]
        )
    else:
        logger.error("Tried to call score on a model with no outputs.")
        raise ValueError("Cannot score model with no outputs.")

    return scores, profiles


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
class LossConfig(abc.ABC):

    """
    Loss configuration base class for keras training

    Args:
        optimizer: configuration for the optimizer to
            compile with the model
    """

    optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )

    @abc.abstractmethod
    def prepare(self, **kwargs):
        """Do any initializing necessary for generating arguments"""
        pass

    @abc.abstractmethod
    def compile(self, model: tf.keras.Model) -> None:
        """Compile a keras model with the loss configuration"""
        pass


@dataclasses.dataclass
class CustomLoss(LossConfig):
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
    normalization: str = "mean_std"
    loss_variables: List[str] = dataclasses.field(default_factory=list)
    metric_variables: List[str] = dataclasses.field(default_factory=list)
    weights: Mapping[str, float] = dataclasses.field(default_factory=dict)
    _fitted: bool = dataclasses.field(init=False, default=False)

    def prepare(self, output_names: List[str], output_samples: List[tf.Tensor]):
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
        for out_varname, sample in zip(output_names, output_samples):
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
class StandardLoss(LossConfig):
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
