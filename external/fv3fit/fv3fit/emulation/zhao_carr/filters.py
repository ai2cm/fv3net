"""Filters to be used with unbatched tensorflow datasets

These should be callables of a single column of data (i.e. no batch dimension)
it must return a scalar boolean.
"""
from typing import Set
import tensorflow as tf
from fv3fit.emulation.types import TensorDict
import dataclasses
import numpy as np


@dataclasses.dataclass
class HighAntarctic:
    high_antarctic_only: bool = True

    @property
    def input_variables(self) -> Set[str]:
        return {"latitude", "surface_air_pressure"}

    def __call__(self, element: TensorDict) -> tf.Tensor:
        lat = element["latitude"] < np.deg2rad(-60)
        ps = element["surface_air_pressure"] < 700e2
        return (ps & lat)[0]
