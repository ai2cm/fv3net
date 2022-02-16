import dataclasses
import numpy as np
import tensorflow as tf
from typing import Optional, Union, Mapping, Hashable


Output = Union[np.ndarray, tf.Tensor]


@dataclasses.dataclass
class Range:
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self):
        if self.min is not None and self.max is not None:
            if self.max <= self.min:
                raise ValueError(
                    f"max value ({self.max}) must be greater than min "
                    f"value ({self.min})."
                )

    def limit_output(self, output_layer: Output):
        if self.min is None and self.max is None:
            return output_layer
        else:
            return self._range_activation(output_layer)

    def _range_activation(self, output: Output) -> Output:
        # Using this instead of ReLU because using both threshold and
        # max_value args yields unexpected results for thresholds < 0.
        x = output
        zeros = tf.zeros_like(x)
        if self.min is not None:
            x = tf.where(tf.math.less(output, self.min), zeros + self.min, x)
            if self.max is not None:
                x = tf.where(
                    tf.math.logical_and(
                        tf.math.greater_equal(output, self.min),
                        tf.math.less(output, self.max),
                    ),
                    output,
                    x,
                )
        if self.max is not None:
            x = tf.where(tf.math.greater_equal(output, self.max), zeros + self.max, x)
        return x


@dataclasses.dataclass
class RangeConfig:
    """Config class limiting output ranges in keras models.
    Limits range by adding a ReLU activation layer after the output layer.

    Attributes:
        ranges: mapping of variable names to be limited to a Range
        containing min/max values.
    """

    ranges: Mapping[Hashable, Range] = dataclasses.field(default_factory=dict)
