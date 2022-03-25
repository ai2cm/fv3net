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

    @classmethod
    def init_testing(cls, input_variables, output_variables) -> "Hyperparameters":
        """Initialize a default model for a given input/output problem"""
        try:
            hyperparameters = cls()
        except TypeError:
            hyperparameters = cls(  # type: ignore
                input_variables=input_variables, output_variables=output_variables
            )
        return hyperparameters
