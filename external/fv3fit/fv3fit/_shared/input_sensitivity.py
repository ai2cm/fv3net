import numpy as np
from typing import Optional, Mapping
from dataclasses import dataclass


RandomForestInputSensitivity = Mapping[str, Mapping[str, np.ndarray]]
JacobianInputSensitivity = Mapping[str, Mapping[str, np.ndarray]]


@dataclass
class InputSensitivity:
    rf_feature_importances: Optional[RandomForestInputSensitivity]
    jacobians: Optional[JacobianInputSensitivity]
