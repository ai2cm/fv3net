from typing import Set
from dataclasses import dataclass
import tensorflow as tf

from fv3fit.emulation.types import TensorDict


def quantize_step(x, y, bins):
    """Quantize two step data

    y == 0 -> 0
    x == y -> 1
    otherwise 2 + bin y-x
    """
    d = y - x
    # can be 0 to n+1 inclusive
    bin_id = tf.searchsorted(bins, d)
    return tf.where(y == 0, 0, tf.where(y == x, 1, bin_id + 2))


def unquantize_step(x, classes, midpoints):
    """Unquantize the data
    """
    # bind_id can be 0 to n+1 inclusive so need to handle boundaries

    bin_id = tf.where(classes > 1, classes - 2, 0)
    d = tf.gather(midpoints, bin_id)
    return tf.where(classes == 0, tf.zeros_like(x), tf.where(classes == 1, x, x + d))


@dataclass
class StepQuantizer:
    before: str
    after: str
    bin_var: str
    bins: tf.Tensor
    values: tf.Tensor

    def forward(self, x) -> tf.Tensor:
        """Return quantization of delta"""
        out = {**x}
        out[self.bin_var] = quantize_step(x[self.before], x[self.after], self.bins)
        return out

    def backward(self, x) -> tf.Tensor:
        """Return floating point from bin_"""
        out = {**x}
        out[self.after] = unquantize_step(x[self.before], x[self.bin_var], self.values)
        return out


@dataclass
class StepQuantizerFactory:
    """Equispaced quantizers"""

    before: str
    after: str
    bin_var: str
    num_bins: int
    min_: float
    max_: float

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        return (requested_names - {self.bin_var}) | {self.before}

    def _fit_bins(self):
        # use equally spaced bins for now sincet this requires no actual fitting
        # on data...maybe quantiles would be better in the future
        bins = tf.linspace(self.min_, self.max_, self.num_bins)
        return tf.cast(bins, tf.float32)

    def build(self, sample: TensorDict) -> StepQuantizer:
        bins = self._fit_bins()
        midpoints = (bins[:-1] + bins[1:]) / 2
        midpoints = tf.concat([bins[0:1], midpoints, bins[-1:]], 0)
        return StepQuantizer(
            before=self.before,
            after=self.after,
            bin_var=self.bin_var,
            bins=bins,
            values=midpoints,
        )
