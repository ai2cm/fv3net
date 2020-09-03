import xarray as xr
from vcm import safe
import abc
from typing import Iterable, Sequence, Hashable
import logging


logger = logging.getLogger(__file__)


class Predictor(abc.ABC):
    """
    Abstract base class for a predictor object, which has a `predict` method
    that takes in a stacked xarray dataset containing variables defined the class's
    `input_variables` attribute with the first dimension being the `sample_dim_name`
    attribute, and returns predictions for the class's `output_variables` attribute.
    Also implements `load` method. Base class for model classes which implement a
    `fit` method as well, but allows creation of predictor classes to be used in
    (non-training) diagnostic and prognostic settings.
    """

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
    ):
        """Initialize the predictor
        
        Args:
            sample_dim_name: name of sample dimension
            input_variables: names of input variables
            output_variables: names of output variables
        
        """

        super().__init__()
        self.sample_dim_name = sample_dim_name
        self.input_variables = input_variables
        self.output_variables = output_variables

    @abc.abstractmethod
    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        pass

    @abc.abstractmethod
    def load(cls, path: str) -> object:
        """Load a serialized model from a directory."""
        pass

    def predict_columnwise(
        self, X: xr.Dataset, sample_dims: Sequence[Hashable]
    ) -> xr.Dataset:
        """Predict on an unstacked xarray dataset

        Args:
            X: the input data
            sample_dims: A list of dimensions over which samples? are taken

        Returns:
            the predictions defined on the same dimensions as X
        """
        coords = X.coords

        inputs_ = safe.get_variables(X, self.input_variables)
        stacked = safe.stack_once(inputs_, "sample", dims=sample_dims)
        transposed = stacked.transpose("sample", ...)
        output = self.predict(transposed).unstack("sample")

        # ensure the output coords are the same
        # stack/unstack adds coordinates if none exist before
        for key in output.coords:
            if key in coords:
                output.coords[key] = coords[key]
            else:
                del output.coords[key]

        # ensure dimension order is the same
        dim_order = [dim for dim in X.dims if dim in output.dims]
        return output.transpose(*dim_order)
