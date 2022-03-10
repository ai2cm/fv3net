import tensorflow as tf
from typing import Optional

from .normalization import NormalizeConfig, DenormalizeConfig


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
            norm_layer:
        """
        super().__init__(*args, **kwargs)

        self._selection = selection

        if normalize is not None:
            self.normalize = NormalizeConfig(
                class_name=normalize,
                layer_name=f"normalized_{self.name}",
                sample_data=sample_in,
            ).initialize_layer()
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
        *args,
        sample_out: Optional[tf.Tensor] = None,
        denormalize: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            nfeatures: size of the output feature dimension
            sample_out: Output sample for variable to fit denormalization layer.
            denormalize: denormalize layer key to use on
                the dense layer output
        """
        super().__init__(*args, **kwargs)

        if denormalize is not None:
            self.denorm = DenormalizeConfig(
                class_name=denormalize,
                layer_name=f"denormalized_{self.name}",
                sample_data=sample_out,
            ).initialize_layer()
        else:
            self.denorm = tf.keras.layers.Lambda(
                lambda x: x, name=f"passthru_{self.name}"
            )

    def call(self, tensor):

        tensor = self.denorm(tensor)

        return tensor


class IncrementStateLayer(tf.keras.layers.Layer):
    """
    Layer for incrementing states with a tendency tensor

    Attributes:
        dt_sec: timestep delta in seconds
    """

    def __init__(self, dt_sec: int, *args, dtype=tf.float32, **kwargs):

        self._dt_sec_arg = dt_sec
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
        dt_sec: int,
        *args,
        sample_in: Optional[tf.Tensor] = None,
        sample_out: Optional[tf.Tensor] = None,
        denormalize: Optional[str] = None,
        tendency_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            sample_in: Input sample for setting denorm layer
            sample_out: Output sample for variable to set shape
                and fit denormalization layer.
            dt_sec: Timestep length in seconds to use for incrementing
                input state.
            denormalize: denormalize layer key to use on
                the dense layer output
            tendency_name: name for the tendency layer otherwise defaults
                to 'tendency_of_{self.name}`
        """
        super().__init__(*args, **kwargs)

        self._dt_sec = dt_sec

        if sample_out is None or sample_in is None:
            tendency_sample = None
        else:
            tendency_sample = (sample_out - sample_in) / dt_sec

        self.tendency = FieldOutput(
            denormalize=denormalize,
            sample_out=tendency_sample,
            name=(
                f"tendency_of_{self.name}" if tendency_name is None else tendency_name
            ),
        )
        self.increment = IncrementStateLayer(dt_sec, name=f"increment_{self.name}")

    def call(self, field_input, network_output):

        tendency = self.tendency(network_output)
        tensor = self.increment(field_input, tendency)

        return tensor

    def get_tendency_output(self, network_output):
        return self.tendency(network_output)
