from dataclasses import dataclass
from copy import copy
import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator


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
        if hasattr(new_regressor, "random_seed"):
            new_regressor.random_seed += len(self.regressors)
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
class BaseXarrayEstimator:
    def fit(
        self, input_vars: tuple, output_vars: tuple, sample_dim: str, data: xr.Dataset
    ):
        """
        Args:
            input_vars: list of input variables
            output_vars: list of output_variables
            sample_dim: dimension over which samples are taken
            data: xarray Dataset with dimensions (sample_dim, *)

        Returns:
            fitted model
        """
        raise NotImplementedError

    def predict(self, data: xr.Dataset, sample_dim: str) -> xr.Dataset:
        """
        Make a prediction

        Args:
            data: xarray Dataset with the same feature dimensions as trained
              data
            sample_dim: dimension along which "samples" are defined. This could be
              inferred, but explicity is not terrible.
        Returns:
            prediction:
        """
        raise NotImplementedError


def _unique_dim_name(data):
    return "".join(data.dims)


def _pack(data: xr.Dataset, sample_dim):
    feature_dim_name = _unique_dim_name(data)
    stacked = data.to_stacked_array(feature_dim_name, sample_dims=[sample_dim])
    return (
        stacked.transpose(sample_dim, feature_dim_name).data,
        stacked.indexes[feature_dim_name],
    )


def _unpack(data: np.ndarray, sample_dim, feature_index):
    da = xr.DataArray(
        data, dims=[sample_dim, "feature"], coords={"feature": feature_index}
    )
    return da.to_unstacked_dataset("feature")


class SklearnWrapper(BaseXarrayEstimator):
    """Wrap a SkLearn model for use with xarray

    """

    def __init__(self, model: BaseEstimator):
        """

        Args:
            model: a scikit learn regression model
        """
        self.model = model

    def __repr__(self):
        return "SklearnWrapper(\n%s)" % repr(self.model)

    def fit(
        self, input_vars: tuple, output_vars: tuple, sample_dim: str, data: xr.Dataset
    ):
        # TODO the sample_dim can change so best to use feature dim to flatten
        self.input_vars_ = input_vars
        self.output_vars_ = output_vars
        x, _ = _pack(data[input_vars], sample_dim)
        y, self.output_features_ = _pack(data[output_vars], sample_dim)
        self.model.fit(x, y)
        return self

    def predict(self, data, sample_dim):
        x, _ = _pack(data[self.input_vars_], sample_dim)
        y = self.model.predict(x)
        ds = _unpack(y, sample_dim, self.output_features_)
        return ds.assign_coords({sample_dim: data[sample_dim]})
