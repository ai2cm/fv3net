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
from typing import Callable, Mapping, Optional, Any
from ..._shared.config import RegularizerConfig

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


def combine_sequence_inputs(inputs: Mapping[str, tf.Tensor]) -> tf.Tensor:
    """Combine inputs into a single (batch, sequence, channels) tensors

    Args:
        inputs: a dictionary of tensors of shape ``(batch, sequence)`` or
            ``(batch, 1)``.  All non-unit values of ``sequence`` must be equal.

    Returns:
        a tensor with shape ``(batch, sequence, len(inputs))``. inputs with
        shape ``(batch, 1)`` are broadcasted to ``(batch, sequence)`` shape and
        stacked along the final dimension.


    """
    list_inputs = [inputs[key] for key in sorted(inputs)]
    expanded_inputs = [tf.expand_dims(tensor, axis=-1) for tensor in list_inputs]
    seq_lengths = set(input.shape[1] for input in expanded_inputs)
    nz = list(seq_lengths - {1})[0]
    broadcasted_inputs = [
        tf.repeat(tensor, nz, 1) if tensor.shape[1] == 1 else tensor
        for tensor in expanded_inputs
    ]
    return tf.concat(broadcasted_inputs, axis=-1)


class HybridRNN(tf.keras.layers.Layer):
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


class RNNBlock(tf.keras.layers.Layer):
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
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
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
        self._kernel_regularizer = kernel_regularizer

        self.dense = [
            tf.keras.layers.Dense(
                width,
                activation=activation,
                kernel_regularizer=self._kernel_regularizer,
            )
            for i in range(depth)
        ]

    def call(self, input: tf.Tensor) -> tf.Tensor:

        outputs = input

        for i in range(len(self.dense)):
            outputs = self.dense[i](outputs)

        return outputs

    def get_config(self):

        config = super().get_config()
        config.update(
            {
                "width": self._width,
                "depth": self._depth,
                "activation": self._activation,
                "kernel_regularizer": self._kernel_regularizer,
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

    def __init__(
        self,
        feature_lengths: Mapping[str, int],
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
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
        self._kernel_regularizer = kernel_regularizer

        self.output_layers = {
            name: tf.keras.layers.Dense(
                feature_length,
                name=f"standard_output_{name}",
                kernel_regularizer=self._kernel_regularizer,
            )
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
        output_channels: Mapping[str, int],
        share_conv_weights: bool = False,
        *args,
        **kwargs,
    ):
        """
        Args:
            feature_lengths: Map of output variable names to expected
                feature dimension length (typically the number of vertical levels)
        """
        super().__init__(*args, **kwargs)
        self._feature_lengths = feature_lengths
        self._single_dense = share_conv_weights

        if share_conv_weights:
            layer_cls = tf.keras.layers.Conv1D
        else:
            layer_cls = tf.keras.layers.LocallyConnected1D

        self.output_layers = {
            name: layer_cls(output_channels.get(name, 1), 1, name=f"rnn_output_{name}")
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
            if field_out.shape[-1] == 1:
                fields[name] = tf.squeeze(field_out, axis=-1)
            else:
                fields[name] = field_out

        return fields


_ARCHITECTURE_KEYS = (
    "rnn-v1-shared-weights",
    "rnn-v1",
    "rnn",
    "dense",
    "linear",
    "dense-local",
)


class _HiddenArchitecture(tf.keras.layers.Layer):
    """Combines an input, architecture and output layer"""

    def __init__(
        self,
        input_layer: Callable[[Mapping[str, tf.Tensor]], tf.Tensor],
        arch_layer: tf.keras.layers.Layer,
        output_layer: tf.keras.layers.Layer,
    ):
        super().__init__()
        self.arch = arch_layer
        self.input_layer = input_layer
        self.outputs = output_layer

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
        kernel_regularizer: configuration of kernel regularization. Only used for
            dense model. Used to initialize RegularizationConfig.
    """

    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    output_channels: Mapping[str, int] = dataclasses.field(default_factory=dict)
    kernel_regularizer: RegularizerConfig = dataclasses.field(
        default_factory=lambda: RegularizerConfig("none")
    )

    def __post_init__(self):
        if self.name not in _ARCHITECTURE_KEYS:
            raise KeyError(f"Unrecognized architecture key: {self.name}")
        if self.name != "dense" and self.kernel_regularizer.instance is not None:
            raise ValueError("Only dense architecture supports kernel regularization.")

    def build(self, feature_lengths: Mapping[str, int]) -> tf.keras.layers.Layer:
        """
        Build the model architecture layer from key

        Args:
            feature_lengths: Map of output variable names to expected
                feature dimension length. used to partition layer outputs
        """
        key = self.name
        kwargs = self.kwargs

        if key == "rnn-v1":
            return _HiddenArchitecture(
                combine_sequence_inputs,
                RNNBlock(**kwargs),
                RNNOutput(
                    feature_lengths,
                    share_conv_weights=True,
                    output_channels=self.output_channels,
                ),
            )
        elif key == "rnn-v1-shared-weights":
            return _HiddenArchitecture(
                combine_sequence_inputs,
                RNNBlock(**kwargs),
                RNNOutput(
                    feature_lengths,
                    share_conv_weights=True,
                    output_channels=self.output_channels,
                ),
            )
        elif key == "rnn":
            return _HiddenArchitecture(
                combine_sequence_inputs,
                HybridRNN(**kwargs),
                StandardOutput(feature_lengths),
            )
        elif key == "dense":
            return _HiddenArchitecture(
                combine_inputs,
                MLPBlock(kernel_regularizer=self.kernel_regularizer.instance, **kwargs),
                StandardOutput(
                    feature_lengths, kernel_regularizer=self.kernel_regularizer.instance
                ),
            )
        elif key == "dense-local":
            return _HiddenArchitecture(
                combine_sequence_inputs,
                MLPBlock(**kwargs),
                RNNOutput(
                    feature_lengths,
                    share_conv_weights=True,
                    output_channels=self.output_channels,
                ),
            )
        elif key == "linear":
            if kwargs:
                raise TypeError("No keyword arguments accepted for linear model")
            return _HiddenArchitecture(
                combine_inputs, MLPBlock(depth=0), StandardOutput(feature_lengths)
            )
