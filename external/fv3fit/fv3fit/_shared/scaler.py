import abc
from copy import copy
import numpy as np
from typing import Mapping, BinaryIO
import xarray as xr

from .packer import ArrayPacker


class NormalizeTransform(abc.ABC):
    @abc.abstractmethod
    def normalize(self, y: np.ndarray):
        pass

    @abc.abstractmethod
    def denormalize(self, y: np.ndarray):
        pass

    @abc.abstractmethod
    def dump(self, f: BinaryIO):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, f: BinaryIO):
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
        packer: ArrayPacker,
        delp_weights: np.ndarray,
        variable_scale_factors: Mapping[str, float] = None,
        sqrt_weights: bool = False,
    ):
        """Weights vertical variables by their relative masses (via delp)
        and upscales variables by optional scale factors.

        Args:
            packer: ArrayPacker object that contains information a
            delp_weights: 1D array of pressure thicknesses for each model level, used to
                scale weights by the levels' relative masses.
            variable_scale_factors: Optional mapping of variable names to scale factors
                by which their weights will be multiplied when normalizing. This allows
                 the mass weighted outputs to be scaled to the same order of magnitude.
                Default of None will target dQ2 features by a factor of 1000.
            sqrt_weights: If True, will square root the weights returned by
                _create_weight_array. Useful if this is used as a target transform
                regressor in fv3fit.sklearn with a MSE loss, as there is no current way
                to directly weight the loss function terms. If set to take sqrt of
                weights in the target transform, the MSE loss function terms will be
                approximately weighted to the layer mass.
        """
        self._variable_scale_factors = variable_scale_factors or {"dQ2": 1000.0}
        delp_weights = np.sqrt(delp_weights) if sqrt_weights is True else delp_weights

        if len(packer.feature_counts) == 0:
            raise ValueError(
                "Packer's feature count information is empty. Make sure the packer has "
                "been packed at least once so that dimension lengths are known."
            )
        self.weights = self._create_weight_array(delp_weights, packer)

    def _create_weight_array(self, delp_weights: np.ndarray, packer: ArrayPacker):
        n_vertical_levels = len(delp_weights)
        weights = {}
        for var in packer.pack_names:
            if packer.feature_counts[var] == n_vertical_levels:
                array = np.reshape(copy(delp_weights), (1, -1))
                dims = [packer.sample_dim_name, f"{var}_feature"]
            elif packer.feature_counts[var] == 1:
                array = np.array([1.0])
                dims = [packer.sample_dim_name]
            else:
                raise ValueError(
                    f"Output variable {var} has {packer.feature_counts[var]} "
                    "features > 1 but not equal to number of vertical levels "
                    f"{n_vertical_levels}."
                )
            if var in self._variable_scale_factors:
                # want to multiply by scale factor when dividing by weights
                array /= self._variable_scale_factors[var]
            weights[var] = (dims, array)
        return packer.to_array(xr.Dataset(weights))  # type: ignore

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
