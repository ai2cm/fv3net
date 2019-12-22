from dataclasses import dataclass

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator


class BatchTransformRegressor:
    """Base class for sklearn-type regressors that are incrementally trained in batches
    So that we can add new estimators at the start of new training batch.

    """

    def __init__(self, transform_regressor):
        self.transform_regressor = transform_regressor
        self.n_estimators_per_batch = self.n_estimators
        self.num_batches_fit = 0

    @property
    def n_estimators(self):
        try:
            return getattr(self.transform_regressor.regressor, "n_estimators")
        except AttributeError:
            try:
                return getattr(
                    self.transform_regressor.regressor.estimator, "n_estimators"
                )
            except AttributeError:
                raise ValueError(
                    "Unable to get number of estimators per regressor."
                    "Check that the regressor is either sklearn "
                    "RandomForestRegressor, or MultiOutputRegressor "
                    "with multiple estimators per regressor"
                )

    def _add_new_batch_estimators(self):
        new_total_estimators = self.n_estimators + self.n_estimators_per_batch
        try:
            setattr(
                self.transform_regressor.regressor.n_estimators, new_total_estimators
            )
        except AttributeError:
            try:
                self.transform_regressor.regressor.set_params(
                    estimator__n_estimators=new_total_estimators
                )
            except ValueError:
                raise ValueError(
                    "Cannot add more estimators to model. Check that model is"
                    "either sklearn RandomForestRegressor "
                    "or MultiOutputRegressor "
                )

    def fit(self, features, outputs):
        if self.num_batches_fit > 0:
            self._add_new_batch_estimators()
        self.transform_regressor.fit(features, outputs)
        self.num_batches_fit += 1


@dataclass
class Packer:
    """Uses: i) getting features out of xarray dataset into np array and
    ii) putting np array of predictions into xrarray dataset

    """

    input_vars: tuple
    output_vars: tuple
    sample_dim: str

    def feature_matrix(self, ds):
        return self.flatten(ds[self.input_vars], self.sample_dim).values

    def target_matrix(self, ds):
        outputs = self.flatten(ds[self.output_vars], self.sample_dim)
        self.output_features_dim_name_ = [
            dim for dim in outputs.dims if dim != self.sample_dim
        ][0]
        self.output_features_ = outputs.indexes[self.output_features_dim_name_]
        return outputs.values

    def ds_prediction(self, prediction_matrix, features, sample_dim):
        """

        Args:
            prediction_matrix: np array of prediction
            features: np.ndarray produced using Packer.flatten on the input data
            sample_dim: name of sample dimension

        Returns:
            xarray dataset containing model prediction
        """
        ds_prediction = xr.DataArray(
            prediction_matrix,
            dims=[sample_dim, "feature"],
            coords={sample_dim: features[sample_dim], "feature": self.output_features_},
        ).to_unstacked_dataset("feature")
        return ds_prediction

    def _remove(self, dims, sample_dim):
        return tuple([dim for dim in dims if dim != sample_dim])

    def _unused_name(self, old_names):
        # should not conflict with existing name
        # a random string that no-one will ever use
        return "dadf3q32d9a09cf"

    def flatten(self, data: xr.Dataset, sample_dim) -> np.ndarray:
        feature_dim_name = self._unused_name(data.dims)
        stacked = data.to_stacked_array(feature_dim_name, sample_dims=[sample_dim])
        return stacked.transpose(sample_dim, feature_dim_name)


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
        self.packer = Packer(input_vars, output_vars, sample_dim)
        features = self.packer.feature_matrix(data)
        targets = self.packer.target_matrix(data)
        self.model.fit(features, targets)

    def predict(self, data, sample_dim):
        features = self.packer.flatten(data, sample_dim)
        prediction = self.model.predict(features.values)
        ds_prediction = self.packer.xr_prediction(prediction, features, sample_dim)
        return ds_prediction
