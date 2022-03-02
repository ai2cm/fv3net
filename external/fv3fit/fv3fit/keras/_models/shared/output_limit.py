import dataclasses
import numpy as np
import tensorflow as tf
from typing import Optional, Union, Mapping, Hashable, Sequence


Output = Union[np.ndarray, tf.Tensor]


@dataclasses.dataclass
class OutputLimit:
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
            return self._limit_activation(output_layer)

    def _limit_activation(self, output: Output) -> Output:
        # Using this instead of ReLU because using threshold args < 0
        # result in a noncontinuous function
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
class OutputLimitConfig:
    """Config class limiting output limits in keras models.
    Limits range by adding piecewise activation for the output layers.

    Attributes:
        limits: mapping of output variable names to be limited by a OutputLimit
        containing min/max values.
    """

    limits: Mapping[Hashable, OutputLimit] = dataclasses.field(default_factory=dict)

    def apply_output_limiters(
        self, outputs: Sequence[Output], names: Sequence[str]
    ) -> Sequence[Output]:
        limited_outputs = []
        for name, output in zip(names, outputs):
            if name in self.limits:
                limited_outputs.append(self.limits[name].limit_output(output))
            else:
                limited_outputs.append(output)
        return limited_outputs
