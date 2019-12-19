from dataclasses import dataclass

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator


def remove(dims, sample_dim):
    return tuple([dim for dim in dims if dim != sample_dim])


def unused_name(old_names):
    # should not conflict with existing name
    # a random string that no-one will ever use
    return "dadf3q32d9a09cf"


def _flatten(data: xr.Dataset, sample_dim) -> np.ndarray:
    feature_dim_name = unused_name(data.dims)
    stacked = data.to_stacked_array(feature_dim_name, sample_dims=[sample_dim])
    return stacked.transpose(sample_dim, feature_dim_name)


class BatchTrainingRegressor:
    """Base class for sklearn-type regressors that are incrementally trained in batches
    So that we can add new estimators at the start of new training batch.

    """
    def __init__(self, regressor):
        self.regressor = regressor
        self.n_estimators_per_batch = regressor.n_estimators

    def add_new_batch_estimators(self):
        if "n_estimators" in self.regressor.__dict__:
            new_total_estimators = \
                self.regressor.n_estimators + self.n_estimators_per_batch
            self.regressor.n_estimators = new_total_estimators
        elif (
            hasattr(self.regressor, "estimator")
            and "n_estimators" in self.regressor.estimator.__dict__
        ):
            self.regressor.set_params(
                estimator__n_estimators=self.regressor.estimator.n_estimators
                + self.n_estimators_per_batch
            )
        else:
            raise ValueError(
                "Cannot add more estimators to model. Check that model is"
                "either sklearn RandomForestRegressor "
                "or MultiOutputRegressor."
            )

    def fit(self, features, outputs):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError


class TransformedTargetRegressor(BatchTrainingRegressor):
    """Modeled off of sklearn's TransformedTargetRegressor but with
    the ability to save the same means/stddev used in normalization without
    having to provide them again to the inverse transform at prediction time.

    """
    def save_normalization_data(self, output_means, output_stddevs):
        self.output_means = output_means
        self.output_stddevs = output_stddevs

    def fit(self, features, outputs):
        normed_outputs = self._transform(outputs)
        self.regressor.fit(features, normed_outputs)

    def predict(self, features):
        normed_outputs = self.regressor.predict(features)
        physical_outputs = self._inverse_transform(normed_outputs)
        return physical_outputs

    def _transform(self, output_matrix):
        """

        Args:
            output_matrix: physical values of targets

        Returns:
            targets normalized by (target-mean) / stddev
        """
        return np.divide(
            np.subtract(output_matrix, self.output_means), self.output_stddevs
        )

    def _inverse_transform(self, output_matrix):
        """

        Args:
            output_matrix: normalized prediction values

        Returns:
            physical values of predictions
        """
        return np.add(
            np.multiply(self.output_stddevs, output_matrix), self.output_means
        )


@dataclass
class BaseXarrayEstimator:
    def fit_xarray(
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

    def predict_xarray(self, data: xr.Dataset, sample_dim: str) -> xr.Dataset:
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


class SklearnWrapper(BaseXarrayEstimator):
    """Wrap a SkLearn model for use with xarray

    """

    def __init__(self, model: BaseEstimator):
        """

        Args:
            model: a scikit learn regression model
        """
        self.model = model
        if "n_estimators" in self.model.__dict__:
            self.n_estimators_per_batch = self.model.n_estimators
        elif (
            hasattr(self.model, "estimator")
            and "n_estimators" in self.model.estimator.__dict__
        ):
            self.n_estimators_per_batch = self.model.estimator.n_estimators

    def __repr__(self):
        return "SklearnWrapper(\n%s)" % repr(self.model)

    def fit(self, features, targets):
        self.model.fit(features, targets)

    def fit_xarray(
        self, input_vars: tuple, output_vars: tuple, sample_dim: str, data: xr.Dataset
    ):
        self.input_vars_ = input_vars
        self.output_vars_ = output_vars
        self.feature_dims_ = remove(data.dims, sample_dim)
        inputs = _flatten(data[input_vars], sample_dim).values
        outputs = _flatten(data[output_vars], sample_dim)

        self.output_features_dim_name_ = [
            dim for dim in outputs.dims if dim != sample_dim
        ][0]
        self.output_features_ = outputs.indexes[self.output_features_dim_name_]
        self.model.fit(inputs, outputs.values)

    def predict(self, features):
        return self.model.predict(features)

    def predict_xarray(self, data, sample_dim):
        inputs = _flatten(data[self.input_vars_], sample_dim).values
        prediction = self.model.predict(inputs)
        ds = xr.DataArray(
            prediction,
            dims=[sample_dim, "feature"],
            coords={sample_dim: inputs[sample_dim], "feature": self.output_features_},
        )
        return ds.to_unstacked_dataset("feature")


