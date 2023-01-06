import dacite
import dataclasses
import fsspec
import io
import joblib
import numpy as np
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


class ReservoirComputingReadout:
    """Readout layer of the reservoir computing model

    linear_regressor: a sklearn Ridge regressor
    square_half_hidden_state: if True, square even terms in the reservoir
        state before it is used as input to the regressor's .fit and
        .predict methods. This option was found to be important for skillful
        predictions in Wikner+2020 (https://doi.org/10.1063/5.0005541)
    """

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

    def dumps(self) -> bytes:
        components = {
            "hyperparameters": dataclasses.asdict(self.hyperparameters),
            "coefficients": self.coefficients,
            "intercepts": self.intercepts,
        }
        f = io.BytesIO()
        joblib.dump(components, f)
        return f.getvalue()


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

    _READOUT_NAME = "readout.pkl"

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
        print(self.coefficients.shape, input.shape, self.intercepts.shape)
        print(self.coefficients.todense())
        print(np.dot(self.coefficients, input))
        print(self.coefficients * input)
        return self.coefficients * input + self.intercepts

    @classmethod
    def load(cls, paths: Sequence[str]) -> "CombinedReservoirComputingReadout":
        """Load a model from a remote path"""
        readouts = []
        for path in paths:
            mapper = fsspec.get_mapper(path)

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
