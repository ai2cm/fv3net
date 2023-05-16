"""zhao carr specific transformations

These will typically depend on the variable zhao_carr used by the zhao carr
microphysics
"""
import dataclasses
from typing import List, Mapping
from fv3fit.emulation.layers.architecture import ArchitectureConfig
from fv3fit.emulation.layers.normalization import MeanMethod, NormFactory, StdDevMethod

import tensorflow as tf
import fv3fit.emulation.transforms.zhao_carr as zhao_carr
from fv3fit.emulation.models.base import Model
from fv3fit.emulation.models.microphysics import MicrophysicsConfig


@dataclasses.dataclass
class PrecpdModelConfig:
    """Configuration for precpd model
    """

    architecture: ArchitectureConfig = ArchitectureConfig(
        name="rnn-v1-shared-weights", kwargs=dict(channels=256)
    )
    precpd_only: bool = True

    def _config(self) -> Model:
        return MicrophysicsConfig(
            input_variables=[
                zhao_carr.T_GSCOND,
                zhao_carr.QV_GSCOND,
                zhao_carr.CLOUD_GSCOND,
                zhao_carr.PrecpdOnly.log_cloud_input,
                zhao_carr.PrecpdOnly.log_humidity_input,
                zhao_carr.DELP,
                zhao_carr.SURFACE_PRESSURE,
                zhao_carr.PRESSURE,
            ],
            direct_out_variables=[
                zhao_carr.PrecpdOnly.qc_diff_scale,
                zhao_carr.PrecpdOnly.qv_diff_scale,
                zhao_carr.PrecpdOnly.t_diff_scale,
            ],
            normalize_default=NormFactory(
                scale=StdDevMethod.all, center=MeanMethod.per_feature
            ),
            timestep_increment_sec=900,
            architecture=self.architecture,
        )

    @property
    def name(self) -> str:
        return "precpd-only"

    @property
    def output_variables(self) -> List[str]:
        return self._config().output_variables

    @property
    def input_variables(self) -> List[str]:
        return self._config().input_variables

    def build(self, data: Mapping[str, tf.Tensor]) -> tf.keras.Model:
        return self._config().build(data)
