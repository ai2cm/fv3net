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
    max_lat: float = -60

    @property
    def input_variables(self) -> Set[str]:
        return {"latitude", "surface_air_pressure"}

    def mask(self, element: TensorDict) -> tf.Tensor:
        lat = element["latitude"] < np.deg2rad(self.max_lat)
        ps = element["surface_air_pressure"] < 700e2
        return ps & lat

    def __call__(self, element: TensorDict) -> tf.Tensor:
        return self.mask(element)[0]
