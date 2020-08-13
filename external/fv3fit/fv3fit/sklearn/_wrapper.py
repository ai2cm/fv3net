from dataclasses import dataclass
from copy import copy
import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator
from .._shared import pack, unpack, Predictor

from typing import Iterable


class RegressorEnsemble:
    """Ensemble of regressors that are incrementally trained in batches

    """

    def __init__(self, base_regressor):
        self.base_regressor = base_regressor
        self.regressors = []

    @property
    def n_estimators(self):
        return len(self.regressors)

    def fit(self, features, outputs):
        """ Adds a base regressor fit on features to the ensemble

        Args:
            features: numpy array of features
            outputs: numpy array of targets

        Returns:

        """
        new_regressor = copy(self.base_regressor)
        # each regressor needs different randomness
        if hasattr(new_regressor, "random_state"):
            new_regressor.random_state += len(self.regressors)
        new_regressor.fit(features, outputs)
        self.regressors.append(new_regressor)

    def predict(self, features):
        """

        Args:
            features: 2D numpy array of features to predict on

        Returns:
            2D numpy array of predictions with N rows corresponding to N input samples.
            Each row is the average ensemble prediction for that sample.
        """
        predictions = np.array(
            [regressor.predict(features) for regressor in self.regressors]
        )
        return np.mean(predictions, axis=0)


@dataclass
class BaseXarrayEstimator(Predictor):
    """Abstract base class for an estimator wich works with xarray datasets
    """

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
    ):
        """
        Args:
            sample_dim_name: dimension over which samples are taken
            input_variables: list of input variables
            output_variables: list of output variables
        """
        super().__init__(sample_dim_name, input_variables, output_variables)

    def fit(self, data: xr.Dataset) -> None:
        """
        Args:
            data: xarray Dataset with dimensions (sample_dim, *)

        """
        raise NotImplementedError


#     def predict(self, data: xr.Dataset, sample_dim: str) -> xr.Dataset:
#         """
#         Make a prediction

#         Args:
#             data: xarray Dataset with the same feature dimensions as trained
#               data
#             sample_dim: dimension along which "samples" are defined. This could be
#               inferred, but explicity is not terrible.
#         Returns:
#             prediction:
#         """
#         raise NotImplementedError


class SklearnWrapper(BaseXarrayEstimator):
    """Wrap a SkLearn model for use with xarray

    """

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: tuple,
        output_variables: tuple,
        model: BaseEstimator,
    ):
        """
        Initialize the wrapper

        Args:
            sample_dim_name: dimension over which samples are taken
            input_variables: list of input variables
            output_variables: list of output variables
            model: a scikit learn regression model
        """
        super().__init__(sample_dim_name, input_variables, output_variables)
        self.model = model

    def __repr__(self):
        return "SklearnWrapper(\n%s)" % repr(self.model)

    def fit(self, data: xr.Dataset):
        # TODO the sample_dim can change so best to use feature dim to flatten
        x, _ = pack(data[self.input_variables], self.sample_dim_name)
        y, self.output_features_ = pack(
            data[self.output_variables], self.sample_dim_name
        )
        self.model.fit(x, y)

    def predict(self, data):
        x, _ = pack(data[self.input_variables], self.sample_dim_name)
        y = self.model.predict(x)
        ds = unpack(y, self.sample_dim_name, self.output_features_)
        return ds.assign_coords({self.sample_dim_name: data[self.sample_dim_name]})
