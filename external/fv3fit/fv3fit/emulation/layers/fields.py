import tensorflow as tf
from typing import Optional

from .normalization import get_norm_class, get_denorm_class


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


class IncrementStateLayer(tf.keras.layers.Layer):
    """
    Layer for incrementing states with a tendency tensor

    Attributes:
        dt_sec: timestep delta in seconds
    """

    def __init__(self, dt_sec: int, *args, dtype=tf.float32, **kwargs):

        self.dt_sec = tf.constant(dt_sec, dtype=dtype)
        super().__init__(*args, **kwargs)

    def call(self, initial: tf.Tensor, tendency: tf.Tensor) -> tf.Tensor:
        """
        Increment state with tendency * timestep

        args:
            tensors: Input state field and corresponding tendency tensor to
                increment by.
        """
        return initial + tendency * self.dt_sec


class IncrementedFieldOutput(tf.keras.layers.Layer):
    """
    Add input tensor to an output tensor using timestep increment.
    This residual-style architecture is analogous to learning tendency-based updates.
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

    def call(self, field_input, network_output):

        tendency = self.tendency(network_output)
        tensor = self.increment(field_input, tendency)

        if self.use_relu:
            tensor = self.relu(tensor)

        return tensor

    def get_tendency_output(self, network_output):
        return self.tendency(network_output)


class CombineInputs(tf.keras.layers.Layer):
    """Input tensor stacking with option to add a dimension for RNNs"""

    def __init__(
        self, combine_axis: int, *args, expand_axis: Optional[int] = None, **kwargs
    ):
        """
        Args:
            combine_axis: Axis to concatenate tensors along.  Note that if expand_axis
                is specified, it is applied before concatenation.  E.g., combine_axis=1
                and expand_axis=1 will concatenate along the newly created dimension.
            expand_axis: New axis to add to the input tensors
        """
        super().__init__(*args, **kwargs)

        self.combine_axis = combine_axis
        self.expand_axis = expand_axis

    def call(self, inputs):

        if self.expand_axis is not None:
            inputs = [
                tf.expand_dims(tensor, axis=self.expand_axis) for tensor in inputs
            ]

        return tf.concat(inputs, axis=self.combine_axis)