from typing import List, Optional
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

FV3CONFIG_FILENAME = "fv3config.yml"


@dataclasses.dataclass
class UserConfig:
    """The top-level object for python runtime configurations

    Attributes:
        diagnostics: list of diagnostic file configurations
        fortran_diagnostics: list of Fortran diagnostic outputs. Currently only used by
            post-processing and so only name and chunks items need to be specified.
        scikit_learn: a machine learning configuration
        nudging: nudge2fine configuration. Cannot be used if any scikit_learn model
            urls are specified.
        step_tendency_variables: variables to compute the tendencies of.
            These could in principle be inferred from the requested diagnostic
            names.
        step_storage_variables: variables to compute the storage of. Needed for certain
            diagnostics.
    """

    diagnostics: List[DiagnosticFileConfig]
    fortran_diagnostics: List[FortranFileConfig]
    scikit_learn: MachineLearningConfig = MachineLearningConfig()
    nudging: Optional[NudgingConfig] = None
    step_tendency_variables: List[str] = dataclasses.field(
        default_factory=lambda: list(
            ("specific_humidity", "air_temperature", "eastward_wind", "northward_wind",)
        )
    )
    step_storage_variables: List[str] = dataclasses.field(
        default_factory=lambda: list(("specific_humidity", "total_water"))
    )


def get_config() -> UserConfig:
    """Open the configurations for this run
    
    .. warning::
        Only valid at runtime
    """
    with open("fv3config.yml") as f:
        config = yaml.safe_load(f)
    return dacite.from_dict(UserConfig, config)


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
