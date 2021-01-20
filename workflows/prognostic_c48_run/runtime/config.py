from typing import List, Optional
import dataclasses
import yaml
import f90nml

import dacite

from runtime.diagnostics.manager import DiagnosticFileConfig
from runtime.steppers.nudging import NudgingConfig
from runtime.steppers.machine_learning import MachineLearningConfig

FV3CONFIG_FILENAME = "fv3config.yml"


@dataclasses.dataclass
class UserConfig:
    diagnostics: List[DiagnosticFileConfig]
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
    with open("fv3config.yml") as f:
        config = yaml.safe_load(f)
    return dacite.from_dict(UserConfig, config)


def get_namelist() -> f90nml.Namelist:
    return f90nml.read("input.nml")
