import tensorflow as tf
from typing import Optional

from ..layers import get_norm_class, get_denorm_class, IncrementStateLayer


class FieldInput(tf.keras.layers.Layer):
    """Normalize input tensor and subselect along feature if specified"""

    def __init__(
        self,
        *args,
        sample_in: Optional[tf.Tensor] = None,
        normalize: Optional[str] = None,
        selection: Optional[slice] = None,
        **kwargs,
    ):
        """
        Args:
            sample_in: sample for fitting normalization layer
            normalize: normalize layer key to use for inputs
            selection: slice selection taken along the feature dimension
                of the input
        """
        super().__init__(*args, **kwargs)

        if normalize is not None:
            self.normalize = get_norm_class(normalize)(name=f"normalized_{self.name}")
            self.normalize.fit(sample_in)
        else:
            self.normalize = tf.keras.layers.Lambda(
                lambda x: x, name=f"passthru_{self.name}"
            )

        self.selection = selection

    def call(self, tensor: tf.Tensor):

        tensor = self.normalize(tensor)
        if self.selection is not None:
            tensor = tensor[..., self.selection]

        return tensor


class FieldOutput(tf.keras.layers.Layer):
    """Connect linear dense output layer and denormalize"""

    def __init__(
        self,
        sample_out: tf.Tensor,
        *args,
        denormalize: Optional[str] = None,
        enforce_positive=False,
        **kwargs,
    ):
        """
        Args:
            sample_out: Output sample for variable to set shape
                and fit denormalization layer.
            denormalize: denormalize layer key to use on
                the dense layer output
            alt_name: alternative name for unscaled and denorm layers
        """
        super().__init__(*args, **kwargs)

        self.unscaled = tf.keras.layers.Dense(
            sample_out.shape[-1], name=f"unscaled_{self.name}"
        )

        if denormalize is not None:
            self.denorm = get_denorm_class(denormalize)(
                name=f"denormalized_{self.name}"
            )
            self.denorm.fit(sample_out)
        else:
            self.denorm = tf.keras.layers.Lambda(lambda x: x)

        self.relu = tf.keras.layers.ReLU()
        self.use_relu = enforce_positive

    def call(self, tensor):

        tensor = self.unscaled(tensor)
        tensor = self.denorm(tensor)

        if self.use_relu:
            tensor = self.relu(tensor)

        return tensor


class ResidualOutput(tf.keras.layers.Layer):
    """
    Add input tensor to an output tensor using timestep increment.
    Gets net to learn tendency-based updates.
    """

    def __init__(
        self,
        sample_out: tf.Tensor,
        dt_sec: int,
        *args,
        denormalize: Optional[str] = None,
        enforce_positive: bool = False,
        **kwargs,
    ):
        """
        Args:
            sample_out: Output sample for variable to set shape
                and fit denormalization layer.
            dt_sec: Timestep length in seconds to use for incrementing
                input state.
            denormalize: denormalize layer key to use on
                the dense layer output
        """
        super().__init__(*args, **kwargs)

        self.tendency = FieldOutput(
            sample_out,
            denormalize=denormalize,
            enforce_positive=False,
            name=f"residual_{self.name}",
        )
        self.increment = IncrementStateLayer(dt_sec, name=f"increment_{self.name}")
        self.use_relu = enforce_positive
        self.relu = tf.keras.layers.ReLU()

    def call(self, tensors):

        field_input, network_output = tensors
        tendency = self.tendency(network_output)
        tensor = self.increment([field_input, tendency])

        if self.use_relu:
            tensor = self.relu(tensor)

        return tensor

    def get_tendency_output(self, network_output):
        return self.tendency(network_output)


class CombineInputs(tf.keras.layers.Layer):
    """Input tensor stacking with option to add a dimension for RNNs"""

    def __init__(self, combine_axis: int, *args, expand: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.combine_axis = combine_axis
        self.expand = expand

    def call(self, inputs):

        if self.expand:
            inputs = [
                tf.expand_dims(tensor, axis=self.combine_axis) for tensor in inputs
            ]

        return tf.concat(inputs, axis=self.combine_axis)


def _combine_initializer(do_combine, axis, expand, name):

    if do_combine:
        layer = CombineInputs(axis, expand=expand)
    else:
        layer = tf.keras.layers.Lambda(lambda x: x, name=f"comb_passthru_{name}")
    return layer


class RNNBlock(tf.keras.layers.Layer):
    """
    RNN connected to an optional MLP for prediction
    
    Combines multiple tensors along a new trailing dimension and
    operates in the reverse on the feature dimension, i.e., from
    the top of the atmosphere to the bottom for microphysics data.
    """

    def __init__(
        self,
        *args,
        channels: int = 256,
        dense_width: int = 256,
        dense_depth: int = 1,
        combine_inputs: bool = True,
        activation: str = "relu",
        **kwargs,
    ):
        """
        Args:
            channels: width of RNN layer
            dense_width: width of MLP layer, only used if dense_depth > 0
            dense_depth: number of MLPlayers connected to RNN output, set to
                0 to disable
            combine_inputs: whether to stack the inputs into a single block
            activation: activation function to use for RNN and MLP
        """
        super().__init__(*args, **kwargs)

        self.combine = _combine_initializer(combine_inputs, -1, True, self.name)
        self.rnn = tf.keras.layers.SimpleRNN(
            channels, activation=activation, go_backwards=True
        )
        self.dense = MLPBlock(
            width=dense_width, depth=dense_depth, combine_inputs=False
        )

    def call(self, inputs):
        combined = self.combine(inputs)
        rnn_out = self.rnn(combined)

        output = self.dense(rnn_out)

        return output


class MLPBlock(tf.keras.layers.Layer):
    """MLP layer for basic NN predictions"""

    def __init__(
        self,
        *args,
        width: int = 256,
        depth: int = 2,
        combine_inputs: bool = True,
        activation: str = "relu",
        **kwargs,
    ):
        """
        Args:
            width: width of MLP layer, only used if dense_depth > 0
            depth: number of MLPlayers, set to 0 to disable
            combine_inputs: whether to stack the inputs into a single block
            activation: activation function to use for RNN and MLP
        """
        super().__init__(*args, **kwargs)

        self.combine = _combine_initializer(combine_inputs, -1, False, self.name)

        self.dense = [
            tf.keras.layers.Dense(width, activation=activation) for i in range(depth)
        ]

    def call(self, inputs):
        combined = self.combine(inputs)
        outputs = combined

        for i in range(len(self.dense)):
            outputs = self.dense[i](outputs)

        return outputs


class LinearBlock(MLPBlock):
    """
    No special prediction architecture, just combine input features
    for connection to outputs.
    """

    def __init__(self, *args, combine_inputs=True, **kwargs):
        """
        Args:
            combine_inputs: whether to stack the inputs into a single block
        """
        super().__init__(*args, combine_inputs=combine_inputs, depth=0, **kwargs)
