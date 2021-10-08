from fv3fit._shared.config import RegularizerConfig
import tensorflow as tf
from typing import Sequence
import dataclasses
import tensorflow_addons as tfa


@dataclasses.dataclass
class ConvolutionalNetwork:
    """
    Attributes:
        output: final output of convolutional network
        hidden_outputs: consecutive outputs of hidden layers in convolutional network
    """

    output: tf.Tensor
    hidden_outputs: Sequence[tf.Tensor]


class TransposeInvariant(tf.keras.constraints.Constraint):
    """Constrains `Conv2D` kernel weights to be unchanged when transposed."""

    def __call__(self, w: tf.Tensor):
        # Conv2D kernels are of shape (kernel_column, kernel_row, channels)
        return tf.scalar_mul(0.5, w + tf.transpose(w, perm=(1, 0, 2)))


@dataclasses.dataclass
class ConvolutionalNetworkConfig:
    """
    Describes how to build a convolutional network.

    The convolutional network consists of some number of convolutional layers applying
    square filters along the x- and y- dimensions (second and third last dimensions),
    followed by a final 1x1 convolutional layer producing the output tensor.

    Attributes:
        filters: number of filters per convolutional layer, equal to
            number of neurons in each hidden layer
        depth: number of convolutional layers, including the final 1x1
            convolutional output layer. Must be greater than or equal to 1.
        kernel_size: width of convolutional filters
        kernel_regularizer: configuration of regularization for hidden layer weights
        gaussian_noise: amount of gaussian noise to add to each hidden layer output
        spectral_normalization: if True, apply spectral normalization to hidden layers
    """

    filters: int = 8
    depth: int = 3
    kernel_size: int = 3
    kernel_regularizer: RegularizerConfig = dataclasses.field(
        default_factory=lambda: RegularizerConfig("none")
    )
    gaussian_noise: float = 0.0
    spectral_normalization: bool = False

    def build(
        self, x_in: tf.Tensor, n_features_out: int, label: str = ""
    ) -> ConvolutionalNetwork:
        """
        Take an input tensor to a convolutional network and return the result of a
        convolutional network's prediction, as tensors.

        Can be used within code that builds a larger neural network. This should
        take in and return normalized values.

        Args:
            x_in: 4+ dimensional input tensor whose last dimension is the feature
                dimension, and second and third last dimensions are horizontal
                dimensions
            n_features_out: dimensionality of last (feature) dimension of output
            config: configuration for convolutional network
            label: inserted into layer names, if this function is used multiple times
                to build one network you must provide a different label each time
        
        Returns:
            tensors resulting from the requested convolutional network, each
            tensor has the same dimensionality as the input tensor but will have fewer
            points along the x- and y-dimensions, due to convolutions
        """
        if self.depth < 1:
            raise ValueError("depth must be at least 1, so we can have an output layer")
        if self.filters == 0:
            raise NotImplementedError(
                "filters=0 causes a floating point exception, "
                "and we haven't written a workaround"
            )
        hidden_outputs = []
        x = x_in
        transpose_invariant = TransposeInvariant()
        for i in range(self.depth - 1):
            if self.gaussian_noise > 0.0:
                x = tf.keras.layers.GaussianNoise(
                    self.gaussian_noise, name=f"gaussian_noise_{label}_{i}"
                )(x)
            hidden_layer = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation=tf.keras.activations.relu,
                data_format="channels_last",
                kernel_regularizer=self.kernel_regularizer.instance,
                name=f"convolutional_{label}_{i}",
                kernel_constraint=transpose_invariant,
            )
            if self.spectral_normalization:
                hidden_layer = tfa.layers.SpectralNormalization(
                    hidden_layer, name=f"spectral_norm_{label}_{i}"
                )
            x = hidden_layer(x)
            hidden_outputs.append(x)
        output = tf.keras.layers.Conv2D(
            filters=n_features_out,
            kernel_size=(1, 1),
            activation="linear",
            data_format="channels_last",
            name=f"convolutional_network_{label}_output",
        )(x)
        return ConvolutionalNetwork(hidden_outputs=hidden_outputs, output=output)
