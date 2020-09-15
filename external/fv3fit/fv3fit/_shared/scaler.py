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
        self._fix_constant_features()

    def _fix_constant_features(self):
        for i, std in enumerate(self.std):
            if std == 0.0:
                self.std[i] = 1.0
                self.mean[i] = 0.0

    def normalize(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler.fit must be called before normalize.")
        return (data - self.mean) / self.std

    def denormalize(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler.fit must be called before denormalize.")
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


class ManualScaler(NormalizeTransform):
    def __init__(self, scales):
        self.scales = scales

    def normalize(self, y: np.ndarray):
        return y * self.scales

    def denormalize(self, y: np.ndarray):
        return y / self.scales

    def dump(self, f: BinaryIO):
        data = {}
        if self.scales is not None:
            data["scales"] = self.scales
        return np.savez(f, **data)

    @classmethod
    def load(cls, f: BinaryIO):
        data = np.load(f)
        scales = data.get("scales")
        scaler = cls(scales)
        return scaler


def get_mass_scaler(
    packer: ArrayPacker,
    delp: np.ndarray,
    variable_scale_factors: Mapping[str, float] = None,
    sqrt_scales: bool = False,
) -> ManualScaler:
    """Creates a ManualScaler that mass weights vertical levels by dividing by
    layer mass. Additional specified variables may be scaled by optional scale factors.
    When variables are normalized using the returned scaler, their resulting loss
    function terms scale as (variable_scale_factor / delp) for 3D variables,
    or (variable_scale_factor) for 2D variables. Unless specified otherwise,
    variable scale factors default to 1.

        Args:
            packer: ArrayPacker object that contains information a
            delp: 1D array of pressure thickness used to mass weight model
                levels.
            variable_scale_factors: Optional mapping of variable names to scale factors
                by which their weights will be multiplied when normalizing. This allows
                the weighted outputs to be scaled to the same order of magnitude.
                Default of None will target dQ2 features by a factor of 1000. All
                other variables have an implicit scale factor of 1.
            sqrt_scales: If True, will square root the scale values used by the
                returned ManualScaler. Useful if this is used as a target transform
                regressor in fv3fit.sklearn with a MSE loss, as there is no current way
                to directly weight the loss function terms. If set to take sqrt of
                scales in the target transform, the MSE loss function terms will be
                approximately weighted to the desired weights.
    """
    scales = _create_scaling_array(packer, delp, variable_scale_factors, sqrt_scales)
    return ManualScaler(scales)


def _create_scaling_array(
    packer: ArrayPacker,
    vertical_scales: np.ndarray,
    variable_scale_factors: Mapping[str, float] = None,
    sqrt_scales: bool = True,
) -> np.ndarray:
    """Creates a set of scale values, such that vertical variables are scaled
    by a specified input set of scales and specified variables are optionally scaled
    by the given scale factors. The resulting scale terms go as
    (variable_scale_factor * vertical_scale / sum(vertical_scales)) for 3D variables,
    or (variable_scale_factor) for 2D variables, such that 2D scalars and 3D vectors
    with the same variable_scale_factor have the same total importance.
    Unless specified otherwise, variable scale factors default to 1.

        Args:
            packer: ArrayPacker object that contains information a
            vertical_scale: 1D array of scales for each model level.
            variable_scale_factors: Optional mapping of variable names to scale factors
                by which their loss scales will be multiplied. This allows
                the scaled outputs to be of the same order of magnitude.
                Default of None will scale target dQ2 features by a factor of 1e6; this
                is chosen such that the features values are of the same order as dQ1
                values when used in the sklearn training (which uses sqrt_scales=True
                and applies the transform to the target variables).
                All other variables have an implicit scale factor of 1.
            sqrt_scales: If True, will square root the scales returned by
                this function. Useful if this is used as a target transform
                regressor in fv3fit.sklearn with a MSE loss, as there is no current way
                to directly weight the loss function terms. If set to take sqrt of
                scales in the target transform, the MSE loss function terms will be
                approximately weighted to the desired scales.
    """
    if len(packer.feature_counts) == 0:
        raise ValueError(
            "Packer's feature count information is empty. Make sure the packer has "
            "been packed at least once so that dimension lengths are known."
        )
    variable_scale_factors = variable_scale_factors or {"dQ2": 1000000.0}
    vertical_scales = vertical_scales / vertical_scales.sum()
    n_vertical_levels = len(vertical_scales)
    scales = {}
    for var in packer.pack_names:
        if packer.feature_counts[var] == n_vertical_levels:
            array = np.reshape(copy(vertical_scales), (1, -1))
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
            array *= variable_scale_factors[var]
        scales[var] = (dims, array)
    scales_array = packer.to_array(xr.Dataset(scales))  # type: ignore
    scales_array = np.sqrt(scales_array) if sqrt_scales else scales_array
    return scales_array
