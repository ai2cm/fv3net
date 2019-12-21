from dataclasses import dataclass

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator


class BatchTrainer:
    """Base class for sklearn-type regressors that are incrementally trained in batches
    So that we can add new estimators at the start of new training batch.

    """

    def __init__(self, regressor):
        self.regressor = regressor
        self.n_estimators_per_batch = self.n_estimators
        self.num_batches_fit = 0

    @property
    def n_estimators(self):
        try:
            return getattr(self.regressor, "n_estimators")
        except AttributeError:
            try:
                return getattr(self.regressor.estimator, "n_estimators")
            except AttributeError:
                raise ValueError("Unable to get number of estimators per regressor."
                                 "Check that the regressor is either sklearn "
                                 "RandomForestRegressor, or MultiOutputRegressor "
                                 "with multiple estimators per regressor")

    def _add_new_batch_estimators(self):
        new_total_estimators = self.n_estimators + self.n_estimators_per_batch
        try:
            setattr(self.regressor.n_estimators, new_total_estimators)
        except AttributeError:
            try:
                self.regressor.set_params(estimator__n_estimators=new_total_estimators)
            except ValueError:
                raise ValueError(
                    "Cannot add more estimators to model. Check that model is"
                    "either sklearn RandomForestRegressor "
                    "or MultiOutputRegressor ")

    def increment_fit(self, features, outputs):
        if self.num_batches_fit > 0:
            self.add_new_batch_estimators()
        self.regressor.fit(features, outputs)
        self.num_batches_fit += 1


class TargetTransformer:
    """Modeled off of sklearn's TransformedTargetRegressor but with
        the ability to save the same means/stddev used in normalization without
        having to provide them again to the inverse transform at prediction time.

    """
    def __init__(self, output_means, output_stddevs):
        self.output_means = output_means
        self.output_stddevs = output_stddevs

    def transform(self, output_matrix):
        """

        Args:
            output_matrix: physical values of targets

        Returns:
            targets normalized by (target-mean) / stddev
        """
        return np.divide(
            np.subtract(output_matrix, self.output_means), self.output_stddevs
        )

    def inverse_transform(self, output_matrix):
        """

        Args:
            output_matrix: normalized prediction values

        Returns:
            physical values of predictions
        """
        return np.add(
            np.multiply(self.output_stddevs, output_matrix), self.output_means
        )


class TransformedBatchRegressor:
    """Modeled off of sklearn's TransformedTargetRegressor but with
    the ability to save the same means/stddev used in normalization without
    having to provide them again to the inverse transform at prediction time.

    """
    def __init__(self, batch_trainer, transformer):
        self.batch_trainer = batch_trainer
        self.transformer = transformer

    def fit(self, features, outputs):
        normed_outputs = self.transformer.transform(outputs)
        self.batch_trainer.increment_fit(features, normed_outputs)

    def predict(self, features):
        normed_outputs = self.batch_trainer.regressor.predict(features)
        physical_outputs = self._inverse_transform(normed_outputs)
        return physical_outputs


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
    return ''.join(data.dims)

def _pack(data: xr.Dataset, sample_dim) -> np.ndarray:
    feature_dim_name = _unique_dim_name(data)
    stacked = data.to_stacked_array(feature_dim_name, sample_dims=[sample_dim])
    return stacked.transpose(sample_dim, feature_dim_name).data, stacked.indexes[feature_dim_name]


def _unpack(data: np.ndarray, sample_dim, feature_index):
    da = xr.DataArray(data, dims=[sample_dim, 'feature'], coords={'feature': feature_index})
    return da.to_unstacked_dataset('feature')

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



