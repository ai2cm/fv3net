import abc
from typing import Iterable
import logging

logger = logging.getLogger(__file__)


class Predictor(abc.ABC):
    '''
    Abstract base class for a predictor object, which has a `predict` method
    that takes in an xarray dataset containing variables defined the class's
    `input_variables` attribute, and returns predictions for the class's
    `output_variables` attribute. Also implements 'dump' and 'load' methods.
    Base class for model classes which implement a `fit` method as well, but allows
    creation of predictor classes to be used in (non-training) diagnostic and
    prognostic settings.
    '''
    
    def __init__(
        self,
        sample_dim_name: str
        input_variables: Iterable[str],
        output_variables: Iterable[str]
    ):
        '''Initialize the predictor
        
        Args:
            sample_dim_name: name of sample dimension
            input_variables: names of input variables
            outpiut_variables: names of output variables
        
        '''
        
        super().__init__()
        self._sample_dim = sample_dim_name
        self.input_variables = input_variables
        self.output_variables = output_variables
        
    @abc.abstractmethod
    def predict(self, X: xr.Dataset) -> xr.Dataset:
        pass
    
    @abc.abstractmethod
    def dump(self, path: str) -> None:
        """Serialize the model to a directory."""
        pass

    @abc.abstractmethod
    def load(self, path: str) -> object:
        """Load a serialized model from a directory."""
        pass
        
        
        