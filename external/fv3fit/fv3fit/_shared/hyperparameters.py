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
    def get_default_model(cls, input_variables, output_variables) -> "Hyperparameters":
        try:
            hyperparameters = cls()
        except TypeError:
            hyperparameters = cls(  # type: ignore
                input_variables=input_variables, output_variables=output_variables
            )
        return hyperparameters
