from typing import Sequence
import tensorflow as tf


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
