import abc
from typing import Sequence


class Hyperparameters(abc.ABC):
    """
    Attributes:
        variables: names of data variables required to train the model
    """

    @property
    @abc.abstractmethod
    def variables(self) -> Sequence[str]:
        pass
