"""The architecture of an ML model

This has the following (haskell-like) pseudocode::

    Input = Mapping[str, tf.Tensor]
    Embedding = tf.Tensor
    Hidden = tf.Tensor
    Output = Mapping[str, tf.Tensor]

    CombineInputs :: Input -> Embedding
    ArchLayer :: Embedding -> Hidden
    OutputLayer :: Hidden -> Output

``ArchitectureConfig.build`` composes these internals into a combined  ``Input
-> Output`` function. It is equivalent to

    (get output) . (get arch) . (get combine inputs)
"""
import dataclasses
import tensorflow as tf
from typing import Mapping, Optional, Any

__all__ = ["ArchitectureConfig"]


def combine_inputs(
    inputs: Mapping[str, tf.Tensor],
    combine_axis: int = -1,
    expand_axis: Optional[int] = None,
) -> tf.Tensor:
    """
        Args:
            inputs: a datastructure of tensors to combine into a single tensor
                (sorted by key)
            combine_axis: Axis to concatenate tensors along.  Note that if expand_axis
                is specified, it is applied before concatenation.  E.g., combine_axis=1
                and expand_axis=1 will concatenate along the newly created dimension.
            expand_axis: New axis to add to the input tensors
    """
    list_inputs = [inputs[key] for key in sorted(inputs)]

    if expand_axis is not None:
        expanded_inputs = [
            tf.expand_dims(tensor, axis=expand_axis) for tensor in list_inputs
        ]
    else:
        expanded_inputs = list_inputs

    return tf.concat(expanded_inputs, axis=combine_axis)


class RNNLayer(tf.keras.layers.Layer):
    """Base class for RNN architectures

    Scalar inputs (with singleton final dimensions) are passed to every element
    of the input sequence like this::

        h[i+1] = rnn(inputs[i], h[i], scalars)

    """

    def input_layer(self, inputs: Mapping[str, tf.Tensor]) -> tf.Tensor:
        list_inputs = [inputs[key] for key in sorted(inputs)]
        expanded_inputs = [tf.expand_dims(tensor, axis=-1) for tensor in list_inputs]
        seq_lengths = set(input.shape[1] for input in expanded_inputs)
        nz = list(seq_lengths - {1})[0]
        broadcasted_inputs = [
            tf.repeat(tensor, nz, 1) if tensor.shape[1] == 1 else tensor
            for tensor in expanded_inputs
        ]
        return tf.concat(broadcasted_inputs, axis=-1)


class HybridRNN(RNNLayer):
    """
    RNN connected to an MLP for prediction

    Layer call expects a 3-D input tensor (sample, z, variable),
    which recurses backwards (top-to-bottom for the physics-param data)
    by default over the z dimension.

    Note: while internal RNN has some vertical directionality,
    this architecture does NOT enforce a strictly downward dependence
    """

    def __init__(
        self,
        *args,
        channels: int = 256,
        dense_width: int = 256,
        dense_depth: int = 1,
        activation: str = "relu",
        go_backwards: bool = True,
        **kwargs,
    ):
        """
        Args:
            channels: width of RNN layer
            dense_width: width of MLP layer, only used if dense_depth > 0
            dense_depth: number of MLPlayers connected to RNN output, set to
                0 to disable
            activation: activation function to use for RNN and MLP
            go_backwards: whether to recurse in the reverse direction
                over the dim 1.  If using wrapper outputs, TOA starts
                at 0, should be false
        """
        super().__init__(*args, **kwargs)
        self._channels = channels
        self._dense_width = dense_width
        self._dense_depth = dense_depth
        self._activation = activation
        self._go_backwards = go_backwards

        self.rnn = tf.keras.layers.SimpleRNN(
            channels, activation=activation, go_backwards=go_backwards
        )
        self.dense = MLPBlock(width=dense_width, depth=dense_depth)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """
        Args:
            input: a 3-d tensor with dims (sample, per_var_features, variables)
        """

        rnn_out = self.rnn(input)

        output = self.dense(rnn_out)

        return output

    def get_output_layer(
        self, feature_lengths: Mapping[str, int]
    ) -> tf.keras.layers.Layer:
        return StandardOutput(feature_lengths)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self._channels,
                "dense_depth": self._dense_depth,
                "dense_width": self._dense_width,
                "activation": self._activation,
                "go_backwards": self._go_backwards,
            }
        )
        return config


class RNNBlock(RNNLayer):
    """
    RNN for prediction that preserves vertical information so
    directional dependence is possible

    Layer call expects a 3-D input tensor (sample, z, variable),
    which recurses backwards (top-to-bottom for the physics-param data)
    by default over the z dimension.
    """

    def __init__(
        self,
        *args,
        channels: int = 256,
        depth: int = 2,
        activation: str = "relu",
        go_backwards: bool = True,
        share_conv_weights: bool = True,
        **kwargs,
    ):
        """
        Args:
            channels: width of RNN layer
            depth: depth of connected RNNs
            activation: activation function to use for RNN
            go_backwards: whether to recurse in the reverse direction
                over the dim 1.  If using wrapper outputs, TOA starts
                at 0, should be false
        """
        super().__init__(*args, **kwargs)
        self._share_conv_weights = share_conv_weights
        self._channels = channels
        self._activation = activation
        self._go_backwards = go_backwards
        self._depth = depth

        self.rnn = [
            tf.keras.layers.SimpleRNN(
                channels, activation=activation, return_sequences=True
            )
            for i in range(self._depth)
        ]

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """
        Args:
            input: a 3-d tensor with dims (sample, per_var_features, variables)

        Returns:
            tensor (sample, features, channels)
        """
        output = None

        # switch recurrent operation to go from TOA to bottom for data in physics space
        if self._go_backwards:
            input = tf.reverse(input, tf.convert_to_tensor([-2]))

        for recurrent_layer in self.rnn:
            output = recurrent_layer(input)
            input = output

        if self._go_backwards:
            output = tf.reverse(output, tf.convert_to_tensor([-2]))

        return output

    def get_output_layer(
        self, feature_lengths: Mapping[str, int]
    ) -> tf.keras.layers.Layer:
        return RNNOutput(feature_lengths, share_conv_weights=self._share_conv_weights)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self._channels,
                "activation": self._activation,
                "go_backwards": self._go_backwards,
                "depth": self._depth,
            }
        )
        return config


class MLPBlock(tf.keras.layers.Layer):
    """MLP layer for basic NN predictions"""

    def __init__(
        self,
        *args,
        width: int = 256,
        depth: int = 2,
        activation: str = "relu",
        **kwargs,
    ):
        """
        Args:
            width: width of MLP layer, only used if dense_depth > 0
            depth: number of MLPlayers, set to 0 to disable
            activation: activation function to use for RNN and MLP
        """
        super().__init__(*args, **kwargs)

        self._width = width
        self._depth = depth
        self._activation = activation

        self.dense = [
            tf.keras.layers.Dense(width, activation=activation) for i in range(depth)
        ]

    def call(self, input: tf.Tensor) -> tf.Tensor:

        outputs = input

        for i in range(len(self.dense)):
            outputs = self.dense[i](outputs)

        return outputs

    def input_layer(self, inputs: Mapping[str, tf.Tensor]) -> tf.Tensor:
        return combine_inputs(inputs, combine_axis=-1, expand_axis=None)

    def get_output_layer(
        self, feature_lengths: Mapping[str, int]
    ) -> tf.keras.layers.Layer:
        return StandardOutput(feature_lengths)

    def get_config(self):

        config = super().get_config()
        config.update(
            {
                "width": self._width,
                "depth": self._depth,
                "activation": self._activation,
            }
        )
        return config


def NoWeightSharingSLP(
    n: int = 1, width: int = 128, activation="relu"
) -> tf.keras.Model:
    """Multi-output model which doesn't share any weights between outputs

    Args:
        n: number of outputs
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(n * width, activation=activation),
            tf.keras.layers.Reshape([n, width]),
            tf.keras.layers.LocallyConnected1D(1, 1),
            tf.keras.layers.Flatten(),
        ]
    )


class StandardOutput(tf.keras.layers.Layer):
    """Uses densely-connected layers w/ linear activation"""

    def __init__(self, feature_lengths: Mapping[str, int], *args, **kwargs):
        """
        Args:
            feature_lengths: Map of output variable names to expected
                feature dimension length
        """
        super().__init__(*args, **kwargs)
        self._feature_lengths = feature_lengths

        self.output_layers = {
            name: tf.keras.layers.Dense(feature_length, name=f"standard_output_{name}")
            for name, feature_length in self._feature_lengths.items()
        }

    def call(self, hidden_output: tf.Tensor) -> Mapping[str, tf.Tensor]:
        """
        Args:
            hidden_output: Network output from hidden layers

        Returns:
            Connected model outputs by variable
        """

        return {
            name: layer(hidden_output) for name, layer in self.output_layers.items()
        }


class RNNOutput(tf.keras.layers.Layer):
    """
    Uses locally-connected layers w/ linear activation to
    retain vertical dependence.

    The output feature dimension is determined by the RNN output recursion dimension
    length.  So to produce full output, the model inputs must contain the full
    vertical dimension.

    E.g., Combined inputs [nsamples, 17, 4] -> RNN network out [nsamples, 17, channels]
        -> connected output [nsamples, 17]

    If an output is a single-level (i.e., feature length of 1)
    then it is connected to the lowest layer containing the full vertical information.
    """

    def __init__(
        self,
        feature_lengths: Mapping[str, int],
        share_conv_weights: bool = False,
        *args,
        **kwargs,
    ):
        """
        Args:
            feature_lengths: Map of output variable names to expected
                feature dimension length
        """
        super().__init__(*args, **kwargs)
        self._feature_lengths = feature_lengths
        self._single_dense = share_conv_weights

        if share_conv_weights:
            layer_cls = tf.keras.layers.Conv1D
        else:
            layer_cls = tf.keras.layers.LocallyConnected1D

        self.output_layers = {
            name: layer_cls(1, 1, name=f"rnn_output_{name}")
            for name in self._feature_lengths.keys()
        }

    def call(self, rnn_outputs: tf.Tensor) -> Mapping[str, tf.Tensor]:
        """
        Args:
            hidden_output: Network output from hidden layers of RNN with dim
                [nsamples, nlev, channels]

        Returns:
            Connected model outputs by variable
        """

        fields = {}
        for name, feature_length in self._feature_lengths.items():
            if feature_length == 1:
                rnn_out = rnn_outputs[..., 0:1, :]
            else:
                rnn_out = rnn_outputs

            field_out = self.output_layers[name](rnn_out)
            field_out = tf.squeeze(field_out, axis=-1)
            fields[name] = field_out

        return fields


_ARCHITECTURE_KEYS = (
    "rnn-v1-shared-weights",
    "rnn-v1",
    "rnn",
    "dense",
    "linear",
)


def _get_arch_layer(key: str, kwargs: Mapping) -> tf.keras.layers.Layer:
    if key == "rnn-v1" or key == "rnn-v1-shared-weights":
        return RNNBlock(**kwargs)
    elif key == "rnn-v1-shared-weights":
        return RNNBlock(share_conv_weights=True, **kwargs)
    elif key == "rnn":
        return HybridRNN(**kwargs)
    elif key == "dense":
        return MLPBlock(**kwargs)
    elif key == "linear":
        if kwargs:
            raise TypeError("No keyword arguments accepted for linear model")
        return MLPBlock(depth=0)


class _HiddenArchitecture(tf.keras.layers.Layer):
    def __init__(
        self,
        key: str,
        net_kwargs: Mapping[str, Any],
        feature_lengths: Mapping[str, int],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.arch = _get_arch_layer(key, net_kwargs)
        self.input_layer = self.arch.input_layer
        self.outputs = self.arch.get_output_layer(feature_lengths)

    def call(self, tensors: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
        combined = self.input_layer(tensors)
        net_output = self.arch(combined)
        return self.outputs(net_output)


@dataclasses.dataclass
class ArchitectureConfig:
    """
    Model architecture configuration to produce a model architecture layer that
    takes in a list of tensors and outputs a dict of un-adjusted outputs by
    name.

    Args:
        name: Name of underlying model architecture to use for the emulator.
            See `get_architecture_cls` for a list of supported layers.
        kwargs: keyword arguments to pass to the initialization
            of the architecture layer
    """

    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.name not in _ARCHITECTURE_KEYS:
            raise KeyError(f"Unrecognized architecture key: {self.name}")

    def build(self, feature_lengths: Mapping[str, int]) -> tf.keras.layers.Layer:
        """
        Build the model architecture layer from key

        Args:
            feature_lengths: Map of output variable names to expected
                feature dimension length. used to partition layer outputs
        """
        return _HiddenArchitecture(self.name, self.kwargs, feature_lengths)
