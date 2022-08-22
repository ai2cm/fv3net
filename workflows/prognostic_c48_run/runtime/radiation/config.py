import dataclasses
from typing import Optional
from runtime.steppers.machine_learning import MachineLearningConfig


@dataclasses.dataclass
class RadiationConfig:
    scheme: str
    input_model: Optional[MachineLearningConfig] = None
    offline: bool = True
