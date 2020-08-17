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


class WeightScaler(NormalizeTransform):
    def __init__(self, weights):
        self.weights = weights

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
        weights = data.get("weights")
        scaler = cls(weights)
        return scaler


def create_weight_array(
    packer: ArrayPacker,
    vertical_weights: np.ndarray,
    variable_scale_factors: Mapping[str, float] = None,
    sqrt_weights: bool = False,
) -> np.ndarray:
    """Weights vertical variables by a specified set of weights
    and upscales variables by optional scale factors.

        Args:
            packer: ArrayPacker object that contains information a
            vertical_weights: 1D array of weights for each model level, used to
                scale weights (normalize will divide by the weight)
            variable_scale_factors: Optional mapping of variable names to scale factors
                by which their weights will be multiplied when normalizing. This allows
                 the weighted outputs to be scaled to the same order of magnitude.
                Default of None will target dQ2 features by a factor of 1000.
            sqrt_weights: If True, will square root the weights returned by
                this function. Useful if this is used as a target transform
                regressor in fv3fit.sklearn with a MSE loss, as there is no current way
                to directly weight the loss function terms. If set to take sqrt of
                weights in the target transform, the MSE loss function terms will be
                approximately weighted to the desired weights.
    """
    if len(packer.feature_counts) == 0:
        raise ValueError(
            "Packer's feature count information is empty. Make sure the packer has "
            "been packed at least once so that dimension lengths are known."
        )
    variable_scale_factors = variable_scale_factors or {"dQ2": 1000.0}
    n_vertical_levels = len(vertical_weights)
    weights = {}
    for var in packer.pack_names:
        if packer.feature_counts[var] == n_vertical_levels:
            array = np.reshape(copy(vertical_weights), (1, -1))
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
        if var in variable_scale_factors:
            # want to multiply by scale factor when dividing by weights
            array /= variable_scale_factors[var]
        weights[var] = (dims, array)
    weights_array = packer.to_array(xr.Dataset(weights))
    weights_array = (
        np.sqrt(weights_array) if sqrt_weights is True else weights_array
    )
    return weights_array  # type: ignore
