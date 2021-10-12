import xarray as xr
import abc
from typing import Hashable, Iterable, Sequence
import logging
import warnings

DATASET_DIM_NAME = "dataset"
logger = logging.getLogger(__file__)


class Predictor(abc.ABC):
    """
    Abstract base class for a predictor object, which has a `predict` method
    that takes in a stacked xarray dataset containing variables defined the class's
    `input_variables` attribute with the first dimension being the sample
    dimension, and returns predictions for the class's `output_variables` attribute.
    Also implements `load` method. Base class for model classes which implement a
    `fit` method as well, but allows creation of predictor classes to be used in
    (non-training) diagnostic and prognostic settings.
    """

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        **kwargs,
    ):
        """Initialize the predictor.
        
        Args:
            input_variables: names of input variables
            output_variables: names of output variables
        """
        super().__init__()
        if len(kwargs.keys()) > 0:
            raise TypeError(
                f"received unexpected keyword arguments: {tuple(kwargs.keys())}"
            )
        self.input_variables = input_variables
        self.output_variables = output_variables

    @abc.abstractmethod
    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""

    @abc.abstractmethod
    def dump(self, path: str) -> None:
        """Serialize to a directory."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "Predictor":
        """Load a serialized model from a directory."""
        pass

    def predict_columnwise(
        self,
        X: xr.Dataset,
        sample_dims: Sequence[Hashable] = (),
        feature_dim: Hashable = None,
    ) -> xr.Dataset:
        """
        Deprecated after models' .predict changed to take unstacked data.
        Will be removed in a following PR.

        Predict on an unstacked xarray dataset

        Args:
            X: the input data
            sample_dims: A list of dimensions over which samples are taken
            feature_dim: If provided, the sample_dims will be inferred from this
                value

        Returns:
            the predictions defined on the same dimensions as X
        """
        warnings.warn(
            "The predict_columnwise method is now deprecated since predictors' "
            "predict methods now work on unstacked data. This will be removed "
            "in the near future.",
            DeprecationWarning,
        )
        return self.predict(X)
