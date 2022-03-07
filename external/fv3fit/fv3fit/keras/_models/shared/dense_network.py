from fv3fit._shared.config import RegularizerConfig
import tensorflow as tf
from typing import Sequence
import dataclasses
import tensorflow_addons as tfa


@dataclasses.dataclass
class DenseNetwork:
    """
    Attributes:
        output: final output of dense network
        hidden_outputs: consecutive outputs of hidden layers in dense network
    """

    output: tf.Tensor
    hidden_outputs: Sequence[tf.Tensor]


@dataclasses.dataclass
class DenseNetworkConfig:
    """
    Attributes:
        width: number of neurons in each hidden layer
        depth: number of hidden layers + 1 (the output layer is a layer)
        kernel_regularizer: configuration of regularization for hidden layer weights
        gaussian_noise: amount of gaussian noise to add to each hidden layer output
        spectral_normalization: if True, apply spectral normalization to hidden layers
    """

    width: int = 8
    depth: int = 3
    kernel_regularizer: RegularizerConfig = dataclasses.field(
        default_factory=lambda: RegularizerConfig("none")
    )
    gaussian_noise: float = 0.0
    spectral_normalization: bool = False

    def build(
        self, x_in: tf.Tensor, n_features_out: int, label: str = ""
    ) -> DenseNetwork:
        """
        Take an input tensor to a dense network and return the result of a dense
        network's prediction, as a tensor.

        Can be used within code that builds a larger neural network. This should
        take in and return normalized values.

        Args:
            x_in: input tensor whose last dimension is the feature dimension
            n_features_out: dimensionality of last (feature) dimension of output
            config: configuration for dense network
            label: inserted into layer names, if this function is used multiple times
                to build one network you must provide a different label each time

        Returns:
            tensor resulting from the requested dense network
        """
        hidden_outputs = []
        x = x_in
        for i in range(self.depth - 1):
            if self.gaussian_noise > 0.0:
                x = tf.keras.layers.GaussianNoise(
                    self.gaussian_noise, name=f"gaussian_noise_{label}_{i}"
                )(x)
            hidden_layer = tf.keras.layers.Dense(
                self.width,
                activation=tf.keras.activations.relu,
                kernel_regularizer=self.kernel_regularizer.instance,
                name=f"hidden_{label}_{i}",
            )
            if self.spectral_normalization:
                hidden_layer = tfa.layers.SpectralNormalization(
                    hidden_layer, name=f"spectral_norm_{label}_{i}"
                )
            x = hidden_layer(x)
            hidden_outputs.append(x)
        output = tf.keras.layers.Dense(
            n_features_out, activation="linear", name=f"dense_network_{label}_output",
        )(x)
        return DenseNetwork(hidden_outputs=hidden_outputs, output=output)
