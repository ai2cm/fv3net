import abc
from typing import Set


class Hyperparameters(abc.ABC):
    """
    Attributes:
        variables: names of data variables required to train the model
    """

    @property
    @abc.abstractmethod
    def variables(self) -> Set[str]:
        pass
