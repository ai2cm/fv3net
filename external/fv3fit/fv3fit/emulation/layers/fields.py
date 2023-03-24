import tensorflow as tf
from typing import Optional
from .normalization import NormFactory


class FieldInput(tf.keras.layers.Layer):
    """Normalize input tensor and subselect along feature if specified"""

    def __init__(
        self,
        *args,
        sample_in: Optional[tf.Tensor] = None,
        normalize: Optional[NormFactory] = None,
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
        self.normalize = (
            None
            if normalize is None
            else normalize.build(sample_in, name=f"normalized_{self.name}")
        )
        self.selection = selection

    def call(self, tensor: tf.Tensor) -> tf.Tensor:

        if self.normalize is not None:
            tensor = self.normalize.forward(tensor)

        if self.selection is not None:
            tensor = tensor[..., self.selection]

        return tensor


class FieldOutput(tf.keras.layers.Layer):
    """Connect linear dense output layer and denormalize"""

    def __init__(
        self,
        sample_out: Optional[tf.Tensor] = None,
        denormalize: Optional[NormFactory] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            sample_out: Output sample for variable to fit denormalization layer.
            denormalize: options for denormalization
        """
        super().__init__(name=name)
        self.normalizer = (
            denormalize.build(sample_out, name=f"denormalized_{name}")
            if denormalize
            else None
        )

    def call(self, tensor):
        return self.normalizer.backward(tensor) if self.normalizer else tensor
