from typing import Set
from fv3fit.emulation.types import TensorDict
import dataclasses
import numpy as np


@dataclasses.dataclass
class HighAntarctic:
    high_antarctic_only: bool = True

    @property
    def input_variables(self) -> Set[str]:
        return {"latitude", "surface_air_pressure"}

    def __call__(self, elm: TensorDict):
        lat = elm["latitude"] < np.deg2rad(-60)
        ps = elm["surface_air_pressure"] < 700e2
        return (ps & lat)[0]
