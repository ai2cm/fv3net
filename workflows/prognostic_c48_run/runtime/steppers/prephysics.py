from typing import Union, Sequence
import dataclasses
from runtime.steppers.machine_learning import MachineLearningConfig
import fv3gfs.util


@dataclasses.dataclass
class PrescriberConfig:
    """Configuration for prescribing states in the model from an external source

    Attributes:
        variables: list variable names to prescribe
        data_source: path to the source of the data to prescribe

    Example::

        PrescriberConfig(
            variables=['']
            data_source=""
        )

    """

    variables: Sequence[str]
    data_source: str


class Prescriber:
    """A pre-physics stepper which obtains prescribed values from an external source
    
        TODO: Implement methods
    """

    net_moistening = "net_moistening"

    def __init__(
        self, config: PrescriberConfig, communicator: fv3gfs.util.Commmunicator
    ):

        self._prescribed_variables: Sequence[str] = list(config.variables)
        self._data_source: str = config.data_source

    def __call__(self, time, state):
        return {}, {}, {}

    def get_diagnostics(self, state, tendency):
        return {}

    def get_momentum_diagnostics(self, state, tendency):
        return {}


@dataclasses.dataclass
class PrephysicsConfig:
    """Configuration of pre-physics computations
    
    Attributes:
        config: can be either a MachineLearningConfig or a
            PrescriberConfig, as these are the allowed pre-physics computations
        
    """

    config: Union[PrescriberConfig, MachineLearningConfig]
