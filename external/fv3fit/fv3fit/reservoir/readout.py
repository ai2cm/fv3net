import dacite
import dataclasses
import fsspec
import io
import joblib
import numpy as np
import re
import scipy.sparse
from sklearn.linear_model import Ridge
from typing import Sequence, Optional

from .config import ReadoutHyperparameters


def _square_evens(v: np.ndarray) -> np.ndarray:
    evens = v[::2]
    odds = v[1::2]
    c = np.empty((v.size,), dtype=v.dtype)
    c[0::2] = evens ** 2
    c[1::2] = odds
    return c


def square_even_terms(v: np.ndarray, axis=1) -> np.ndarray:
    return np.apply_along_axis(func1d=_square_evens, axis=axis, arr=v)


def _extract_int_from_subdir(s):
    nums = re.findall(r"\d+", s.rstrip("/").split("/")[-1])
    if len(nums) == 1:
        return int(nums[0])
    else:
        raise ValueError(
            "Ordering of readout subdirectories should be indicated "
            "by a single numeric tag or suffix. ex. 'subdir_0' or '0'."
            f"Subdir {s} violates this naming rule."
        )


def _sort_subdirs_numerically(subdirs: Sequence[str]) -> Sequence[str]:
    """
    Sort a list of subdirs by their numeric tags.
    ex. "subdir_0", "subdir_1"
    """
    nums = [_extract_int_from_subdir(s) for s in subdirs]
    if len(subdirs) != len(np.unique(nums)):
        raise ValueError(
            "Multiple readout subdirectories have the same " "numbering label."
        )
    return [subdir for _, subdir in sorted(zip(nums, subdirs))]


class ReservoirComputingReadout:
    """Readout layer of the reservoir computing model

    hyperparameters: hyperparameters describing the readout
    coefficients: if provided from an already-fit readout,
        use as the linear regression coefficients
    intercepts: if provided from an already-fit readout,
        use as the linear regression intercepts
    """

    _READOUT_NAME = "readout.bin"

    def __init__(
        self,
        hyperparameters: ReadoutHyperparameters,
        coefficients: Optional[np.ndarray] = None,
        intercepts: Optional[np.ndarray] = None,
    ):
        self.hyperparameters = hyperparameters
        self.coefficients = coefficients
        self.intercepts = intercepts
        self.square_half_hidden_state = hyperparameters.square_half_hidden_state

    def fit(self, res_states: np.ndarray, output_states: np.ndarray) -> None:
        if self.coefficients or self.intercepts:
            raise ValueError(
                "Readout has already been fit and has coefficients and intercept "
                "values. Fit method can only be called if readout is yet fit."
            )
        linear_regressor = Ridge(**self.hyperparameters.linear_regressor_kwargs)
        if self.square_half_hidden_state is True:
            res_states = square_even_terms(res_states, axis=1)
        linear_regressor.fit(res_states, output_states)
        self.coefficients = linear_regressor.coef_
        self.intercepts = linear_regressor.intercept_

    def predict(self, input: np.ndarray):
        if self.square_half_hidden_state:
            input = square_even_terms(input, axis=0)
        return np.dot(self.coefficients, input) + self.intercepts

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)
        mapper = fs.get_mapper(path)
        components = {
            "hyperparameters": dataclasses.asdict(self.hyperparameters),
            "coefficients": self.coefficients,
            "intercepts": self.intercepts,
        }
        f = io.BytesIO()
        joblib.dump(components, f)
        mapper[self._READOUT_NAME] = f.getvalue()

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingReadout":
        mapper = fsspec.get_mapper(path)
        f = io.BytesIO(mapper[cls._READOUT_NAME])
        readout_components = joblib.load(f)
        readout_hyperparameters = dacite.from_dict(
            ReadoutHyperparameters, readout_components.pop("hyperparameters")
        )
        return cls(hyperparameters=readout_hyperparameters, **readout_components)


class CombinedReservoirComputingReadout:
    """Combines readout layers of multiple reservoir computing models
    into a block diagonal readout layer, which can be used to predict over
    the combined domain of those models.
    Prediction is done on vector of concatenated reservoir states.

    linear_regressors: sequence of sklearn Ridge regressors, each corresponding
        to a subdomain
    square_half_hidden_state: if True, square even terms in the reservoir state
        before it is used as input to the regressor's .fit and .predict methods
        This option was found to be important for skillful predictions in
        Wikner+2020 (https://doi.org/10.1063/5.0005541)
    """

    _READOUT_NAME = "readout.bin"

    def __init__(self, readouts: Sequence[ReservoirComputingReadout]):
        self._combine_readouts(readouts)

    def _combine_readouts(self, readouts: Sequence[ReservoirComputingReadout]):
        coefs, intercepts, square_state_settings = [], [], []
        for readout in readouts:
            coefs.append(readout.coefficients)
            intercepts.append(readout.intercepts)
            square_state_settings.append(readout.square_half_hidden_state)

        # Merge the coefficient arrays of individual readouts into single
        # block diagonal matrix
        self.coefficients = scipy.sparse.block_diag(coefs)

        # Concatenate the intercepts of individual readouts into single array
        self.intercepts = np.concatenate(intercepts)

        if len(np.unique(square_state_settings)) != 1:
            raise ValueError(
                "All readouts must have the same setting for square_half_hidden_state."
            )
        self.square_half_hidden_state = square_state_settings[0]

    def predict(self, input: np.ndarray):
        if self.square_half_hidden_state:
            input = square_even_terms(input, axis=0)
        return self.coefficients * input + self.intercepts

    @classmethod
    def load(cls, path: str) -> "CombinedReservoirComputingReadout":
        """Load a model from a remote directory. Each subdir in path
        refers to the full reservoir model directory containing the saved
        readout to load within a subdir named "readout". ex.
        | path
        | -- model_0
        |    -- readout.bin
        |    -- reservoir
        | -- model_1
        |    -- readout.bin
              ...
        Assumes model subdirs are numbered in the order that they should
        be used in the combined readout.

        path: directory containing model subdirectories for each readout
        """
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        subdirs = _sort_subdirs_numerically(fs.ls(path))
        if "gs" in fs.protocol:
            subdirs = [f"gs://{path}" for path in subdirs]

        readouts = []

        for subdir in subdirs:
            mapper = fsspec.get_mapper(subdir)

            f = io.BytesIO(mapper[cls._READOUT_NAME])
            readout_components = joblib.load(f)
            hyperparameters = dacite.from_dict(
                ReadoutHyperparameters, readout_components.pop("hyperparameters")
            )
            readouts.append(
                ReservoirComputingReadout(
                    hyperparameters=hyperparameters, **readout_components
                )
            )
        return cls(readouts)
