import abc
from copy import copy
import numpy as np
import xarray as xr
import fsspec
import joblib
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from .._shared import pack, unpack, Predictor

from typing import Iterable


class AppendPrincipalComponents:
    def __init__(self, base_regressor, n_components: int = None, random_state=None):
        """Initialize an AppendPrincipalComponents

        Args:
            base_regressor: regressor being wrapped
            n_components: number of principal components to append, by default
                append all components
            random_state: random state or seed
        """
        self.base_regressor = base_regressor
        self.pca: PCA = PCA(n_components=n_components, random_state=random_state)
        self._is_fit = False

    def fit(self, features: np.ndarray, outputs: np.ndarray) -> None:
        if not self._is_fit:
            self.pca.fit(features)
            self._is_fit = True
        features = self._append_principal_components(features)
        self.base_regressor.fit(features, outputs)

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = self._append_principal_components(features)
        return self.base_regressor.predict(features)

    @property
    def n_estimators(self) -> int:
        return self.base_regressor.n_estimators

    def _append_principal_components(self, features: np.ndarray) -> np.ndarray:
        """Appends the principal components as new features.

        Args:
            features: 2d numpy array [samples, features] to predict on

        Returns:
            features: input array with new principal component features appended
        """
        if self.pca.n_components > 0:
            principal_components = self.pca.transform(features)
            return np.concatenate([features, principal_components], axis=1)
        else:
            return features


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


class BaseXarrayEstimator(Predictor):
    """Abstract base class for an estimator wich works with xarray datasets.
    Subclasses Predictor abstract base class but adds `fit` method taking in
    xarray dataset.
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

    @abc.abstractmethod
    def fit(self, data: xr.Dataset) -> None:
        """
        Args:
            data: xarray Dataset with dimensions (sample_dim, *)

        """
        raise NotImplementedError


class SklearnWrapper(BaseXarrayEstimator):
    """Wrap a SkLearn model for use with xarray

    """

    _MODEL_FILENAME = "sklearn_model.pkl"

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
        self._sample_dim_name = sample_dim_name
        self._input_variables = input_variables
        self._output_variables = output_variables
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

    @classmethod
    def load(cls, path: str) -> Predictor:
        """Load a wrapped model saved as a .pkl file specified by `path`"""
        fs, _, _ = fsspec.get_fs_token_paths(path)
        with fs.open(path, "rb") as f:
            wrapped_model = joblib.load(f)
        return wrapped_model

    # these are here for backward compatibility with pre-unified API attribute names
    @property
    def input_variables(self):
        if hasattr(self, "_input_variables"):
            return self._input_variables
        elif hasattr(self, "input_vars_"):
            return self.input_vars_
        else:
            raise ValueError("Wrapped model version without input variables attribute.")

    @property
    def output_variables(self):
        if hasattr(self, "_input_variables"):
            return self._output_variables
        elif hasattr(self, "input_vars_"):
            return self.output_vars_
        else:
            raise ValueError(
                "Wrapped model version without output variables attribute."
            )

    @property
    def sample_dim_name(self):
        return getattr(self, "_sample_dim_name", "sample")
