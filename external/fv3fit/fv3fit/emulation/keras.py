import abc
import dataclasses
import logging
import tensorflow as tf
from typing import Any, Optional, Mapping, List, Sequence, Tuple, Union

from fv3fit.emulation.layers.normalization import NormalizeConfig
from .scoring import score_multi_output, score_single_output
from .._shared.config import OptimizerConfig

logger = logging.getLogger(__name__)


def save_model(model: tf.keras.Model, destination: str):

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

    prediction = model.predict(inputs)

    if len(model.output_names) > 1:
        scores, profiles = score_multi_output(
            targets, prediction, model.output_names
        )
    elif len(model.output_names) == 1:
        scores, profiles = score_single_output(
            targets, prediction, model.output_names[0]
        )
    else:
        logger.error("Tried to call score on a model with no outputs.")
        raise ValueError("Cannot score model with no outputs.")

    return scores, profiles


class NormalizedMSE(tf.keras.losses.MeanSquaredError):
    def __init__(self, norm_cls_name, sample_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._normalize = NormalizeConfig(norm_cls_name, sample_data).initialize_layer()

    def call(self, y_true, y_pred):
        return super().call(self._normalize(y_true), self._normalize(y_pred))


@dataclasses.dataclass
class KerasCompileArgs(abc.ABC):

    optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )

    @abc.abstractmethod
    def prepare(self, **kwargs):
        """Do any initializing necessary for generating arguments"""
        pass

    @abc.abstractmethod
    def get(self) -> Mapping[str, Any]:
        """Provide a keyword dictionary with arguments for keras model compilation"""
        pass


@dataclasses.dataclass
class CustomKerasCompileArgs(KerasCompileArgs):
    normalization = "mean_std"
    loss_variables: List[str] = dataclasses.field(default_factory=list)
    metric_variables: List[str] = dataclasses.field(default_factory=list)
    weights: Mapping[str, float] = dataclasses.field(default_factory=dict)
    _fitted: bool = dataclasses.field(init=False, default=False)

    def prepare(self, output_names: List[str], output_samples: List[tf.Tensor]):
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

    def get(self) -> Mapping[str, Any]:
        return {
            "loss": self._loss,
            "metrics": self._metrics,
            "weights": self._weights,
            "optimizer": self.optimizer.instance,
        }


KerasMetrics = List[str]
KerasWeights = Union[Mapping[str, float], List[float]]


@dataclasses.dataclass
class StandardKerasCompileArgs(KerasCompileArgs):
    loss: Optional[str] = None
    metrics: Optional[KerasMetrics] = None
    weights: Optional[KerasWeights] = None

    def prepare(self, **kwargs):
        """Nothing to do here"""
        pass

    def get(self) -> Mapping[str, Any]:
        return {
            "loss": self.loss,
            "metrics": self.metrics,
            "weights": self.weights,
            "optimizer": self.optimizer.instance,
        }
