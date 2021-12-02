from typing import Mapping, Optional, Sequence, Union
import tensorflow as tf


class CombineInputs(tf.keras.layers.Layer):
    """Input tensor stacking with option to add a dimension for RNNs"""

    def __init__(
        self, *args, combine_axis: int = -1, expand_axis: Optional[int] = None, **kwargs
    ):
        """
        Args:
            combine_axis: Axis to concatenate tensors along.  Note that if expand_axis
                is specified, it is applied before concatenation.  E.g., combine_axis=1
                and expand_axis=1 will concatenate along the newly created dimension.
            expand_axis: New axis to add to the input tensors
        """
        super().__init__(*args, **kwargs)

        self._combine_axis = combine_axis
        self._expand_axis = expand_axis

    def _call_with_tensors(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:

        if self._expand_axis is not None:
            inputs = [
                tf.expand_dims(tensor, axis=self._expand_axis) for tensor in inputs
            ]

        return tf.concat(inputs, axis=self._combine_axis)

    def call(self, inputs: Union[Sequence[tf.Tensor], Mapping[str, tf.Tensor]]):
        if isinstance(inputs, Mapping):
            return self._call_with_tensors([inputs[key] for key in sorted(inputs)])
        else:
            return self._call_with_tensors(inputs)

    def get_config(self):

        config = super().get_config()
        config.update(
            {"combine_axis": self._combine_axis, "expand_axis": self._combine_axis}
        )
        return config


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

    def call(self, input):
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


class RNN(tf.keras.layers.Layer):
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
        self._channels = channels
        self._activation = activation
        self._go_backwards = go_backwards
        self._depth = depth

        self.rnn = [
            tf.keras.layers.SimpleRNN(
                channels, activation=activation,
                return_sequences=True
            )
            for i in range(self._depth)
        ]

    def call(self, input):
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

    def call(self, input):

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


class StandardOutputConnector(tf.keras.layers.Layer):
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
            name: tf.keras.layers.Dense(
                feature_length,
                name=f"standard_output_{name}"
            )
            for name, feature_length in self._feature_lengths.items()
        }

    def call(self, hidden_output) -> Mapping[str, tf.Tensor]:
        """
        Args:
            hidden_output: Network output from hidden layers

        Returns:
            Connected model outputs by variable
        """

        return {
            name: layer(hidden_output)
            for name, layer in self.output_layers.items()
        }


class RNNOutputConnector(tf.keras.layers.Layer):
    """
    Uses locally-connected layers w/ linear activation to
    retain vertical dependence.

    The output feature dimension is determined by the RNN output recursion dimension length.  So to produce full output, the model inputs must contain the full
    vertical dimension.

    E.g., Combined inputs [nsamples, 17, 4] -> RNN network out [nsamples, 17, channels] ->
        connected output [nsamples, 17]

    If an output is a single-level (i.e., feature length of 1)
    then it is connected to the lowest layer containing the full vertical information.
    """

    def __init__(self, feature_lengths: Mapping[str, int], *args, **kwargs):
        """
        Args:
            feature_lengths: Map of output variable names to expected
                feature dimension length
        """
        super().__init__(*args, **kwargs)
        self._feature_lengths = feature_lengths

        self.output_layers = {
            name: tf.keras.layers.LocallyConnected1D(1, 1, name=f"rnn_output_{name}")
            for name in self._feature_lengths.keys()
        }

    def call(self, rnn_outputs) -> Mapping[str, tf.Tensor]:
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
