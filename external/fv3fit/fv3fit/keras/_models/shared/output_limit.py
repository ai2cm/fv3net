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


@dataclasses.dataclass
class OutputSquashConfig:
    squash_threshold: Optional[float] = None
    squash_to: Optional[float] = None
    squash_by_name: Optional[str] = None
    squash_on_train: bool = False

    """Config class squashing outputs in keras models.
    Will squash all outputs to a specified target (e.g. 0.) based on a threshold for
    a specific output.

    Attributes:
        squash_threshold: value in the specified output which will determine
        whether outputs are squashed
        squash_to: value to which values will be squashed
        squash_by_name: name of the output to which the threshold will be applied;
        must be present in the outputs and this output must be broadcastable to
        every output
        squash_on_train: if true, apply squashing on during model training; otherwise,
        apply only on prediction
    """

    def __post_init__(self):
        if self.squash_by_name is not None:
            if self.squash_threshold is None or self.squash_to is None:
                raise ValueError(
                    "If squash_by_name is specified, squash_threshold and squash_to "
                    "must be also."
                )

    def squash_outputs(
        self, names: Sequence[str], outputs: Sequence[Output]
    ) -> Sequence[Output]:
        if self.squash_by_name is not None:
            if self.squash_by_name not in names:
                raise ValueError(
                    f"The squash by variable ({self.squash_by_name}) must among the "
                    "set of output names."
                )
            squashed_outputs = []
            squash_by_output = outputs[names.index(self.squash_by_name)]
            for output in outputs:
                squashed_outputs.append(self._squash_output(output, squash_by_output))
            return squashed_outputs
        else:
            return outputs

    def _squash_output(self, target_output: Output, squash_by_output: Output) -> Output:
        x = target_output
        squashed = tf.constant(self.squash_to, dtype=np.float32) * tf.ones_like(
            x, dtype=np.float32
        )
        x = tf.where(tf.math.less(squash_by_output, self.squash_threshold), squashed, x)
        return x
