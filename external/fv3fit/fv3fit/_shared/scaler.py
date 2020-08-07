import abc
import numpy as np
from typing import Sequence, Mapping, Union, BinaryIO


class NormalizeTransform(abc.ABC):
    @abc.abstractmethod
    def fit():
        pass

    @abc.abstractmethod
    def normalize(y: np.ndarray):
        pass
    
    @abc.abstractmethod
    def denormalize(y: np.ndarray):
        pass

    @abc.abstractmethod
    def dump():
        pass

    @abc.abstractmethod
    def load():
        pass


class StandardScaler(NormalizeTransform):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std = data.std(axis=0).astype(np.float32)

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean

    def dump(self, f: BinaryIO):
        data = {}
        if self.mean is not None:
            data["mean"] = self.mean
        if self.std is not None:
            data["std"] = self.std
        return np.savez(f, **data)

    @classmethod
    def load(cls, f: BinaryIO):
        data = np.load(f)
        scaler = cls()
        scaler.mean = data.get("mean")
        scaler.std = data.get("std")
        return scaler


class MassScaler(NormalizeTransform):
    def __init__(self):
        self.weights = None

    def fit(
            self,
            output_var_order: Sequence[str],
            output_var_feature_count: Mapping[str, int],
            delp_weights: np.ndarray,
            variable_scale_factors: Mapping[str, float] = None,
            sqrt_weights: bool = False
    ):
        """Weights vertical variables by their relative masses (via delp)
        and upscales variables by optional scale factors.

        Args:
            output_var_order: Sequence of target variable names
            output_var_feature_count: Dict with number of features in each variable.
                2D variables have count 1, 3D should have number of features equal
                to number of vertical levels.
            delp_weights: 1D array of pressure thicknesses for each model level, used to
                scale weights by the levels' relative masses.
            variable_scale_factors: Optional mapping of variable names to scale factors
                by which their weights will be multiplied when normalizing. This allows the mass
                weighted outputs to be scaled to the same order of magnitude.
                Default of None will target dQ2 features by a factor of 1000.
            sqrt_weights: If True, will square root the weights returned by
                _create_weight_array. Useful if this is used as a target transform
                regressor in fv3fit.sklearn with a MSE loss, as there is no current way
                to directly weight the loss function terms. If set to take sqrt of
                weights in the target transform, the MSE loss function terms will be
                approximately weighted to the layer mass.
        """
        self._variable_scale_factors = variable_scale_factors or {"dQ2": 1000.}
        self.weights = self._create_weight_array(
            delp_weights, output_var_order, output_var_feature_count)
        if sqrt_weights:
            self.weights = np.sqrt(self.weights)

    def _create_weight_array(
            self,
            delp_weights: np.ndarray,
            output_var_order: Sequence[str],
            output_var_feature_count: Mapping[str, int]):
        n_levels = len(delp_weights)
        weights = np.array([])
        for var in output_var_order:
            n_features = output_var_feature_count[var]
            if n_features == n_levels:
                var_weights = delp_weights
            elif n_features == 1:
                var_weights = np.array([1.])
            else:
                raise ValueError(
                    f"Output variable {var} has {n_features} features > 1 "
                    f"but not equal to number of vertical levels {n_levels}.")
            if var in self._variable_scale_factors:
                # want to multiply by scale factor when dividing by weights
                var_weights /= self._variable_scale_factors[var]
            np.append(weights, var_weights)
        return weights
    
    def normalize(self, y: np.ndarray):
        return y / self.weights

    def denormalize(self, y: np.ndarray):
        return y * self.weights

    def dump(self, f: BinaryIO):
        data = {}
        if self.weights is not None:
            data["weights"] = self.weights
        return np.savez(f, **data)

    @classmethod
    def load(cls, f: BinaryIO):
        data = np.load(f)
        scaler = cls()
        scaler.weights = data.get("weights")
        return scaler
    

        