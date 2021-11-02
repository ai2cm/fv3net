from fv3fit._shared.config import RegularizerConfig
import tensorflow as tf
from typing import Sequence
import dataclasses


@dataclasses.dataclass
class LocallyConnectedNetwork:
    """
    Attributes:
        hidden_outputs: consecutive outputs of hidden layers in convolutional network
    """

    hidden_outputs: Sequence[tf.Tensor]


@dataclasses.dataclass
class LocallyConnectedNetworkConfig:
    """
    Describes how to build a locally-connected (in the feature dimension) network.

    Input for all outputs is the full feature space, but weights are devoted
    independently to each output feature when kernel_size=1.

    Attributes:
        filters: number of filters per layer, analogous to number of neurons
            in a dense layer being applied at a single level
        depth: number of locally-connected layers, including the final output layer
            and the initial Convolution2D layer (to take a full field of inputs).
            Must be greater than or equal to 2.
        kernel_size: width of filters, if greater than 1 will mean hidden features
            are shared across nearby feature levels
        kernel_regularizer: configuration of regularization for hidden layer weights
        activation_function: name of keras activation function to use on hidden layers
        implementation: implementation argument to pass to LocallyConnected1D,
            generally should use 1 for large dense networks (many filters,
            big kernel size), 2 for small networks, 3 for large sparse networks
            (many filters, small kernel size), only affects RAM requirement and
            performance, not results
    """

    filters: int = 32
    depth: int = 3
    kernel_size: int = 3
    kernel_regularizer: RegularizerConfig = dataclasses.field(
        default_factory=lambda: RegularizerConfig("none")
    )
    gaussian_noise: float = 0.0
    activation_function: str = "relu"
    implementation: int = 2

    def __post_init__(self):
        if self.depth < 2:
            raise ValueError(
                "depth must be at least 2, so we can have "
                "a convolutional and output layer"
            )
        if self.filters == 0:
            raise NotImplementedError(
                "filters=0 causes a floating point exception, "
                "and we haven't written a workaround"
            )

    def build(
        self, x_in: tf.Tensor, n_features_out: int, label: str = ""
    ) -> LocallyConnectedNetwork:
        """
        Take an input tensor to a locally-connected network and return the result of a
        locally-connected network's prediction, as tensors.

        Can be used within code that builds a larger neural network. This should
        take in and return normalized values.

        Args:
            x_in: 4+ dimensional input tensor whose last dimension is the feature
                dimension (generally the vertical dimension), and second and third
                last dimensions are horizontal dimensions
            n_features_out: dimensionality of last (feature) dimension of output
            label: inserted into layer names, if this function is used multiple times
                to build one network you must provide a different label each time
        
        Returns:
            tensors resulting from the requested locally-connected network
        """
        hidden_outputs = []
        if self.implementation == 1 and self.kernel_size > 1:
            raise ValueError("for implementation=1, must have kernel_size of 1")
        x = x_in
        if self.gaussian_noise > 0.0:
            x = tf.keras.layers.GaussianNoise(
                self.gaussian_noise, name=f"gaussian_noise_{label}_initial"
            )(x)
        hidden_layer = tf.keras.layers.Convolution2D(
            filters=self.filters * n_features_out,
            kernel_size=(1, 1),
            padding="same",
            activation=self.activation_function,
            data_format="channels_last",
            kernel_regularizer=self.kernel_regularizer.instance,
            name=f"locally_connected_{label}_initial_conv2d",
        )
        x = hidden_layer(x)
        # shape is now [batch, x, y, self.filters * vertical]
        # needs to be [batch-like, vertical, filters]
        x = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, (-1, n_features_out, self.filters)),
            output_shape=(n_features_out, self.filters),
        )(x)
        hidden_outputs.append(x)

        for i in range(self.depth - 2):
            if self.gaussian_noise > 0.0:
                x = tf.keras.layers.GaussianNoise(
                    self.gaussian_noise, name=f"gaussian_noise_{label}_{i}"
                )(x)
            hidden_layer = tf.keras.layers.LocallyConnected1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding=self._padding,
                activation=self.activation_function,
                data_format="channels_last",
                kernel_regularizer=self.kernel_regularizer.instance,
                name=f"locally_connected_{label}_{i}",
                implementation=self.implementation,
            )
            x = hidden_layer(x)
            hidden_outputs.append(x)

        return LocallyConnectedNetwork(hidden_outputs=hidden_outputs)

    @property
    def _padding(self):
        if self.implementation == 1:
            padding = "valid"  # valid is identical to same if kernel_size is 1
        else:
            padding = "same"
        return padding

    def get_output(
        self,
        x_in: tf.Tensor,
        x_hidden: tf.Tensor,
        name: str,
        n_features_out: int,
        label: str = "",
    ) -> tf.Tensor:
        # it would be great to not depend on a private attribute here,
        # but didn't find another way to get the true shape
        shape_in = tf.shape(x_in)._inferred_value
        x = tf.keras.layers.FixedLocallyConnected1D(
            filters=1,
            kernel_size=self.kernel_size,
            padding=self._padding,
            activation="linear",
            data_format="channels_last",
            name=f"locally_connected_{label}_{name}_output",
            implementation=self.implementation,
        )(x_hidden)
        x = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, (-1, shape_in[1], shape_in[2], n_features_out)),
            output_shape=(shape_in[1], shape_in[2], n_features_out),
        )(x)
        return x
