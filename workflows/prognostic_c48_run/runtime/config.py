from typing import List, Optional, Union, Iterable
import dataclasses
import yaml
import f90nml

import dacite

from runtime.diagnostics.manager import (
    DiagnosticFileConfig,
    FortranFileConfig,
    get_chunks,
)
from runtime.steppers.nudging import NudgingConfig
from runtime.steppers.machine_learning import MachineLearningConfig
from runtime.steppers.prescriber import PrescriberConfig
from runtime.transformers.tendency_prescriber import TendencyPrescriberConfig
import runtime.transformers.emulator

FV3CONFIG_FILENAME = "fv3config.yml"
FV3CONFIG_KEYS = {
    "namelist",
    "experiment_name",
    "diag_table",
    "data_table",
    "field_table",
    "initial_conditions",
    "forcing",
    "orographic_forcing",
    "patch_files",
    "gfs_analysis_data",
}


@dataclasses.dataclass
class UserConfig:
    """The top-level object for python runtime configurations

    Attributes:
        diagnostics: list of diagnostic file configurations
        fortran_diagnostics: list of Fortran diagnostic file configurations
        prephysics: optional configuration of computations prior to physics,
            specified by either a machine learning configuation or a prescriber
            configuration
        scikit_learn: a machine learning configuration
        nudging: nudge2fine configuration. Cannot be used if any scikit_learn model
            urls are specified.
        tendency_prescriber: configuration for overriding physics tendencies.
    """

    diagnostics: List[DiagnosticFileConfig] = dataclasses.field(default_factory=list)
    fortran_diagnostics: List[FortranFileConfig] = dataclasses.field(
        default_factory=list
    )
    prephysics: Optional[Union[PrescriberConfig, MachineLearningConfig]] = None
    scikit_learn: Optional[MachineLearningConfig] = None
    nudging: Optional[NudgingConfig] = None
    tendency_prescriber: Optional[TendencyPrescriberConfig] = None
    online_emulator: Optional[runtime.transformers.emulator.Config] = None

    @property
    def diagnostic_variables(self) -> Iterable[str]:
        for diag_file_config in self.diagnostics:
            for variable in diag_file_config.variables:
                yield variable


def get_config() -> UserConfig:
    """Open the configurations for this run
    
    .. warning::
        Only valid at runtime
    """
    with open("fv3config.yml") as f:
        config = yaml.safe_load(f)
    runtime_config = {key: config[key] for key in config if key not in FV3CONFIG_KEYS}
    return dacite.from_dict(UserConfig, runtime_config, dacite.Config(strict=True))


def get_namelist() -> f90nml.Namelist:
    """Open the fv3 namelist
    
    .. warning::
        Only valid at runtime
    """
    return f90nml.read("input.nml")


def write_chunks(config: UserConfig):
    """Given UserConfig, write chunks to 'chunks.yaml'"""
    diagnostic_file_configs = (
        config.fortran_diagnostics + config.diagnostics  # type: ignore
    )
    chunks = get_chunks(diagnostic_file_configs)
    with open("chunks.yaml", "w") as f:
        yaml.safe_dump(chunks, f)
