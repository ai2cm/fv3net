import dataclasses
from fv3fit.emulation.layers.normalization import NormalizeConfig
import tensorflow as tf
from typing import Optional, Mapping, List, Union

from tensorflow.keras import optimizers

from .._shared.config import OptimizerConfig


def save_model(model: tf.keras.Model, destination: str):

    model.compiled_loss = None
    model.compiled_metrics = None
    model.optimizer = None
    model_path = os.path.join(destination, "model.tf")
    model.save(model_path, save_format="tf")

    return model_path


class NormalizedMSE(tf.keras.losses.MeanSquaredError):
    def __init__(self, norm_cls_name, sample_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._normalize = NormalizeConfig(norm_cls_name, sample_data).initialize_layer()

    def call(self, y_true, y_pred):
        return super().call(self._normalize(y_true), self._normalize(y_pred))


@dataclasses.dataclass
class CustomKerasLossConfig:
    normalization = "mean_std"
    loss_variables: List[str] = dataclasses.field(default_factory=list)
    metric_variables: List[str] = dataclasses.field(default_factory=list)
    weights: Mapping[str, float] = dataclasses.field(default_factory=dict)
    optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    _fitted: bool = dataclasses.field(init=False, default=False)

    def fit(self, output_names: List[str], output_samples: List[tf.Tensor]):
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

        self.loss = losses
        self.metrics = metrics
        self.weights = weights
        self._fitted = True

    def compile(self, model):
        
        if not self._fitted:
            raise ValueError("CustomKerasLoss requires fit() to be called with sample data prior to compilation.")
        
        model.compile(
            loss=self.loss,
            metrics=self.metrics,
            weights=self.weights,
            optimizer=self.optimizer.instance
        )

        return model


KerasMetrics = List[str]
KerasWeights = Union[Mapping[str, float], List[float]]


@dataclasses.dataclass
class KerasLossConfig:
    loss: Optional[str] = None
    metrics: Optional[KerasMetrics] = None
    weights: Optional[KerasWeights] = None
    optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    _fitted: bool = dataclasses.field(init=False, default=True)

    def fit(self, *args):
        pass

    def compile(self, model):
        model.compile(
            loss=self.loss,
            metrics=self.metrics,
            weights=self.weights,
            optimizer=self.optimizer.instance
        )

        return model


@dataclasses.dataclass
class KerasTrainer:

    epochs: int = 1
    batch_size: int = 128
    valid_freq: int = 5
    verbose: int = 2
    shuffle: Optional[int] = 100_000
    history: tf.keras.callbacks.History = dataclasses.field(init=False)

    def batch_fit(
        self,
        model: tf.keras.Model,
        data: tf.data.Dataset,
        valid_data: Optional[tf.data.Dataset] = None,
        valid_freq: int = 2,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ):

        if callbacks is None:
            callbacks = []

        if self.shuffle is not None:
            data = data.shuffle(self.shuffle)

        if valid_data is not None:
            valid_data = valid_data.batch(self.batch_size)

        self.history = model.fit(
            data.batch(self.batch_size),
            epochs=self.epochs,
            validation_data=valid_data,
            validation_freq=valid_freq,
            verbose=self.verbose,
            callbacks=callbacks,
        )