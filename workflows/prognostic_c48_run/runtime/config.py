from typing import Dict, List, Optional, Mapping
import dataclasses
import yaml
import f90nml

from runtime.diagnostics.manager import DiagnosticFileConfig
from runtime.steppers.nudging import NudgingConfig
from runtime.steppers.machine_learning import MachineLearningConfig

FV3CONFIG_FILENAME = "fv3config.yml"


@dataclasses.dataclass
class UserConfig:
    diagnostics: List[DiagnosticFileConfig]
    scikit_learn: MachineLearningConfig = MachineLearningConfig()
    nudging: Optional[NudgingConfig] = None
    namelist: Mapping = dataclasses.field(default_factory=dict)
    base_version: str = "v0.5"
    step_tendency_variables: List[str] = dataclasses.field(
        default_factory=lambda: list(
            ("specific_humidity", "air_temperature", "eastward_wind", "northward_wind",)
        )
    )
    step_storage_variables: List[str] = dataclasses.field(
        default_factory=lambda: list(("specific_humidity", "total_water"))
    )


def get_config() -> Dict:
    with open("fv3config.yml") as f:
        config = yaml.safe_load(f)
    return config


def get_namelist() -> f90nml.Namelist:
    return f90nml.read("input.nml")
