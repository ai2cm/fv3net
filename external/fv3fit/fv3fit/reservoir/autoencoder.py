import dataclasses
import numpy as np
import os
import tensorflow as tf
from toolz.functoolz import curry
from typing import Union, Sequence, Optional, List, Set, Tuple
from fv3fit._shared import (
    get_dir,
    put_dir,
    register_training_function,
    OptimizerConfig,
    io,
    DatasetPredictor,
)
from fv3fit._shared.training_config import Hyperparameters

from fv3fit.keras import (
    train_pure_keras_model,
    CallbackConfig,
    TrainingLoopConfig,
    LossConfig,
    full_standard_normalized_input,
    standard_denormalize,
    ClipConfig,
    OutputLimitConfig,
)


@io.register("dense-autoencoder")
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

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            encoder_filename = os.path.join(path, self._ENCODER_NAME)
            self.encoder.save(encoder_filename)
            decoder_filename = os.path.join(path, self._DECODER_NAME)
            self.decoder.save(decoder_filename)

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


@dataclasses.dataclass
class DenseAutoencoderHyperparameters(Hyperparameters):
    """
    state_variables: variables to encode/decode
    latent_dim_size: size of latent space that encoder transforms to
    units: number of units in each dense layer
    n_dense_layers: number of dense layers in *each* of encoder/decoder
    loss: configuration of loss functions, will be applied separately to
        each output variable.
    training_loop: configuration of training loop
    optimizer_config: selection of algorithm to be used in gradient descent
    callback_config: configuration for keras callbacks
    normalization_fit_samples: number of samples
        to use when fitting normalization
    output_limit_config: configuration for limiting output values.

    """

    state_variables: Sequence[str]
    latent_dim_size: int = 10
    units: int = 20
    n_dense_layers: int = 2
    loss: LossConfig = LossConfig(scaling="standard", loss_type="mse")
    training_loop: TrainingLoopConfig = dataclasses.field(
        default_factory=TrainingLoopConfig
    )
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    callbacks: List[CallbackConfig] = dataclasses.field(default_factory=list)
    normalization_fit_samples: int = 500_000
    output_limit_config: OutputLimitConfig = dataclasses.field(
        default_factory=lambda: OutputLimitConfig()
    )

    @property
    def variables(self) -> Set[str]:
        return set(self.state_variables)

    @classmethod
    def init_testing(
        cls, input_variables, output_variables
    ) -> "DenseAutoencoderHyperparameters":
        """Initialize a default instance for tests"""
        return cls(state_variables=input_variables)


def build_concat_and_scale_only_autoencoder(
    variables: Sequence[str], X: Sequence[np.ndarray]
) -> tf.keras.Model:
    """ Performs input concatenation and norm/denormalization,
    but does not train actual encoder or decoder layers. Useful for
    reservoir training tests.

    Args:
        X: data for normalization
    """
    input_layers = [
        tf.keras.layers.Input(shape=arr.shape[-1], name=f"{input_name}_input")
        for input_name, arr in zip(variables, X)
    ]
    full_input = full_standard_normalized_input(input_layers, X, variables,)
    encoder = tf.keras.Model(inputs=input_layers, outputs=full_input)

    decoder_input = tf.keras.layers.Input(
        shape=full_input.shape[-1], name="decoder_input"
    )
    decoder_outputs = []
    start_ind = 0
    for output in X:
        decoder_outputs.append(
            decoder_input[..., slice(start_ind, start_ind + output.shape[-1])],
        )
        start_ind += output.shape[-1]
    denorm_output_layers = standard_denormalize(
        names=variables, layers=decoder_outputs, arrays=X,
    )
    decoder = tf.keras.Model(inputs=decoder_input, outputs=denorm_output_layers)

    model = Autoencoder(encoder=encoder, decoder=decoder)
    # Need to call custom model once so it has forward pass information before saving
    model(X)
    return model


def build_autoencoder(
    config: DenseAutoencoderHyperparameters,
    X: Tuple[tf.Tensor, ...],
    y: Tuple[tf.Tensor, ...],
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Args:
        config: configuration of convolutional training
        X: example input for keras fitting, used to determine shape and normalization
        y: example output for keras fitting, used to determine shape and normalization
    """
    input_layers = [
        tf.keras.layers.Input(shape=arr.shape[-1], name=f"{input_name}_encoder_input")
        for input_name, arr in zip(config.state_variables, X)
    ]
    full_input = full_standard_normalized_input(
        input_layers, X, config.state_variables,
    )
    encoder_layers = full_input
    for i in range(config.n_dense_layers):
        encoder_layers = tf.keras.layers.Dense(
            config.units, activation="relu", name=f"encoder_{i}"
        )(encoder_layers)
    encoder_layers = tf.keras.layers.Dense(config.latent_dim_size, activation="relu")(
        encoder_layers
    )
    encoder = tf.keras.Model(inputs=input_layers, outputs=encoder_layers)

    decoder_input = tf.keras.layers.Input(
        shape=config.latent_dim_size, name="decoder_input"
    )
    decoder_layers = decoder_input
    for i in range(config.n_dense_layers):
        decoder_layers = tf.keras.layers.Dense(
            units=config.units, activation="relu", name=f"decoder_{i}"
        )(decoder_layers)

    norm_output_layers = [
        tf.keras.layers.Dense(
            array.shape[-1], activation="linear", name=f"dense_network_output_{i}",
        )(decoder_layers)
        for i, array in enumerate(y)
    ]
    denorm_output_layers = standard_denormalize(
        names=config.state_variables, layers=norm_output_layers, arrays=y,
    )

    # apply output range limiters
    denorm_output_layers = config.output_limit_config.apply_output_limiters(
        outputs=denorm_output_layers, names=config.state_variables
    )

    decoder = tf.keras.Model(inputs=decoder_input, outputs=denorm_output_layers)
    train_model = Autoencoder(encoder=encoder, decoder=decoder)
    predict_model = Autoencoder(encoder=encoder, decoder=decoder)

    stddevs = [
        np.std(array, axis=tuple(range(len(array.shape) - 1)), dtype=np.float32)
        for array in y
    ]
    train_model.compile(
        optimizer=config.optimizer_config.instance,
        loss=[config.loss.loss(std) for std in stddevs],
    )

    # need to call prediction model once so it can save without compiling
    predict_model([np.ones((3, arr.shape[-1])) for arr in X])

    return train_model, predict_model


@register_training_function("dense_autoencoder", DenseAutoencoderHyperparameters)
def train_dense_autoencoder(
    hyperparameters: DenseAutoencoderHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> DatasetPredictor:

    pure_keras = train_pure_keras_model(
        train_batches=train_batches,
        validation_batches=validation_batches,
        build_model=curry(build_autoencoder)(config=hyperparameters),
        input_variables=hyperparameters.state_variables,
        output_variables=hyperparameters.state_variables,
        clip_config=ClipConfig(),
        training_loop=hyperparameters.training_loop,
        build_samples=hyperparameters.normalization_fit_samples,
        callbacks=hyperparameters.callbacks,
    )

    return DatasetPredictor(
        pure_keras.input_variables,
        pure_keras.output_variables,
        model=pure_keras.model,
        unstacked_dims=pure_keras._unstacked_dims,
        n_halo=pure_keras._n_halo,
    )
