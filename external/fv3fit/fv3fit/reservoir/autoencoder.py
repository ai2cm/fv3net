import dataclasses
import numpy as np
import os
import tensorflow as tf
from typing import Union, Sequence, Optional, List, Set
from fv3fit._shared import get_dir, put_dir, register_training_function, OptimizerConfig
from fv3fit._shared.training_config import Hyperparameters
from .. import Predictor
from .domain import concat_variables_along_feature_dim
from fv3fit.keras import CallbackConfig, TrainingLoopConfig, LossConfig, PureKerasModel

from fv3fit.emulation.layers import StandardNormLayer, StandardDenormLayer
from functools import partial


class Autoencoder(tf.keras.Model):
    _ENCODER_NAME = "encoder.tf"
    _DECODER_NAME = "decoder.tf"

    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model):
        super(Autoencoder, self).__init__()
        self.n_latent_dims = encoder.layers[-1].output.shape[-1]
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
        return self.encoder.predict(x)

    def decode(self, latent_x: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
        return self.decoder.predict(latent_x)

    @classmethod
    def load(cls, path: str) -> "Autoencoder":
        with get_dir(path) as model_path:
            encoder = tf.keras.models.load_model(
                os.path.join(model_path, cls._ENCODER_NAME)
            )
            decoder = tf.keras.models.load_model(
                os.path.join(model_path, cls._DECODER_NAME)
            )
        return cls(encoder=encoder, decoder=decoder)

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            encoder_filename = os.path.join(path, self._ENCODER_NAME)
            self.encoder.save(encoder_filename)
            decoder_filename = os.path.join(path, self._DECODER_NAME)
            self.decoder.save(decoder_filename)


@dataclasses.dataclass
class DenseAutoencoderHyperparameters(Hyperparameters):
    """
    input_variables: variables to encode/decode
    output_variables: variables to encode/decode, must match
        input_variables
    latent_dim_size: size of latent space that encoder transforms to
    units: number of units in each dense layer
    n_dense_layers: number of dense layers in *each* of encoder/decoder
    loss: configuration of loss functions, will be applied separately to
        each output variable.
    training_loop: configuration of training loop
    optimizer_config: selection of algorithm to be used in gradient descent
    callback_config: configuration for keras callbacks
    """

    input_variables: Sequence[str]
    output_variables: Sequence[str]
    latent_dim_size: int
    units: int
    n_dense_layers: int
    loss: LossConfig = LossConfig(scaling="standard", loss_type="mse")
    training_loop: TrainingLoopConfig = dataclasses.field(
        default_factory=TrainingLoopConfig
    )
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    callbacks: List[CallbackConfig] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.input_variables != self.output_variables:
            raise ValueError(
                f"Output variables {self.output_variables} must match "
                f"input variables {self.input_variables}."
            )

    @property
    def variables(self) -> Set[str]:
        return set(self.input_variables)

    @property
    def output_varaibles(self) -> Sequence[str]:
        return self.input_variables


def _get_Xy_dataset(data, variables):
    concat_data = data.map(partial(concat_variables_along_feature_dim, variables))
    return tf.data.Dataset.zip((concat_data, concat_data))


def _build_autoencoder(
    sample_data: tf.Tensor,
    input_size: int,
    latent_dim_size: int,
    units: int,
    n_dense_layers: int,
) -> Autoencoder:

    encoder_layers = [
        tf.keras.layers.Input(shape=(input_size), name="input"),
    ]
    norm = StandardNormLayer(name="standard_normalize")
    norm.fit(sample_data)
    encoder_layers.append(norm)
    for i in range(n_dense_layers):
        encoder_layers.append(
            tf.keras.layers.Dense(units, activation="relu", name=f"encoder_{i}")
        )
    encoder_layers.append(tf.keras.layers.Dense(latent_dim_size, activation="relu"))
    encoder = tf.keras.Sequential(encoder_layers)

    decoder_layers = [
        tf.keras.layers.Input(shape=(latent_dim_size), name="decoder_input"),
    ]
    for i in range(n_dense_layers):
        decoder_layers.append(
            tf.keras.layers.Dense(units, activation="relu", name=f"decoder_{i}")
        )
    decoder_layers.append(
        tf.keras.layers.Dense(input_size, activation="linear", name="decoded_output")
    )
    denorm = StandardDenormLayer(name="standard_denormalize")
    denorm.fit(sample_data)
    decoder_layers.append(denorm)

    decoder = tf.keras.Sequential(decoder_layers)

    return Autoencoder(encoder=encoder, decoder=decoder)


@register_training_function("dense-autoencoder", DenseAutoencoderHyperparameters)
def train_dense_autoencoder(
    hyperparameters: DenseAutoencoderHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> Predictor:
    norm_batch = next(iter(train_batches))
    norm_batch_concat = concat_variables_along_feature_dim(
        variables=hyperparameters.input_variables, variable_tensors=norm_batch
    )
    stddev = np.std(norm_batch_concat, axis=0)

    callbacks = [callback.instance for callback in hyperparameters.callbacks]
    train_Xy = _get_Xy_dataset(train_batches, hyperparameters.input_variables)
    if validation_batches is not None:
        val_Xy = _get_Xy_dataset(validation_batches, hyperparameters.input_variables)
    else:
        val_Xy = None

    n_features = norm_batch_concat.shape[-1]
    train_model = _build_autoencoder(
        sample_data=norm_batch_concat,
        input_size=n_features,
        latent_dim_size=hyperparameters.latent_dim_size,
        units=hyperparameters.units,
        n_dense_layers=hyperparameters.n_dense_layers,
    )
    train_model.compile(
        optimizer=hyperparameters.optimizer_config.instance,
        loss=hyperparameters.loss.loss(stddev.astype(np.float32)),
    )
    train_loop = hyperparameters.training_loop
    train_loop.fit_loop(
        model=train_model,
        Xy=train_Xy,
        validation_data=val_Xy,
        callbacks=[callback.instance for callback in callbacks],
    )
    predictor = PureKerasModel(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.input_variables,
        model=train_model,
        unstacked_dims=("z",),
        n_halo=0,
    )
    return predictor
