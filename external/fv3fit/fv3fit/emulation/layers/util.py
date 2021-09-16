from typing import Sequence
import tensorflow as tf


class IncrementStateLayer(tf.keras.layers.Layer):
    """
    Layer for incrementing states with a tendency tensor

    Attributes:
        dt_sec: timestep delta in seconds
    """

    def __init__(self, dt_sec: int, *args, **kwargs):

        self.dt_sec = dt_sec
        super().__init__(*args, **kwargs)

    def call(self, tensors: Sequence[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Increment state with tendency * timestep

        args:
            tensors: Input state field and corresponding tendency tensor to
                increment by.
        """
        initial, tend = tensors
        return initial + tend * self.dt_sec


class PassThruLayer(tf.keras.layers.Layer):
    """
    Layer that passes a tensor unchanged.  Useful for naming/renaming outputs.
    """

    def call(self, tensor):
        return tensor
