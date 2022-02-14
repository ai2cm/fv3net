import numpy as np
from typing import Optional, Mapping, Sequence
from dataclasses import dataclass


# Jacobian format: {y_name: {x_name: dy/dx}}
JacobianInputSensitivity = Mapping[str, Mapping[str, np.ndarray]]


@dataclass
class RandomForestInputSensitivity:
    """
    Attributes:
        mean_importances: average value of feature importance across ensemble
        std_importances: standard deviation of feature importance across ensemble
        indices: for input features of length >1 along feature dimension, the indices
            corresponding to position along feature dimension. Has length of 1 with
            nan index for scalar features.
    """

    mean_importances: Sequence[float]
    std_importances: Sequence[float]
    indices: Sequence[int]


RandomForestInputSensitivities = Mapping[str, RandomForestInputSensitivity]


@dataclass
class InputSensitivity:
    rf_feature_importances: Optional[RandomForestInputSensitivities] = None
    jacobians: Optional[JacobianInputSensitivity] = None
