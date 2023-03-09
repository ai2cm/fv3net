import fsspec
import numpy as np
import os
import scipy.sparse
from typing import Sequence

from .config import BatchLinearRegressorHyperparameters


class NotFittedError(Exception):
    def __init__(self, message):
        super().__init__(message)


def _square_evens(v: np.ndarray) -> np.ndarray:
    evens = v[::2]
    odds = v[1::2]
    c = np.empty((v.size,), dtype=v.dtype)
    c[0::2] = evens ** 2
    c[1::2] = odds
    return c


def square_even_terms(v: np.ndarray, axis=1) -> np.ndarray:
    return np.apply_along_axis(func1d=_square_evens, axis=axis, arr=v)


class BatchLinearRegressor:
    """ Solves for weights W in matrix equation AW=B
    Where A = X_T.X + l2*I and B= X_T.y
    """

    def __init__(self, hyperparameters: BatchLinearRegressorHyperparameters):
        self.hyperparameters = hyperparameters
        self.A = None
        self.B = None

    def _add_bias_feature(self, X):
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    def _check_X_last_col_constant(self, X):
        last_col = X[:, -1]
        if not np.allclose(np.unique(last_col), np.array([1.0])):
            raise ValueError(
                "Last column of X array must all be ones if add_bias_term is False."
            )

    def batch_update(self, X, y):
        if self.hyperparameters.add_bias_term:
            X = self._add_bias_feature(X)
        else:
            self._check_X_last_col_constant(X)

        if self.A is None and self.B is None:
            self.A = np.zeros((X.shape[1], X.shape[1]))
            self.B = np.zeros((X.shape[1], y.shape[1]))

        self.A = np.add(self.A, (np.dot(X.T, X)))
        self.B = np.add(self.B, np.dot(X.T, y))

    def get_weights(self):
        # use_least_squares_solve is useful for simple test cases
        # where np.linalg.solve encounters for singular XT.X

        reg = self.hyperparameters.l2 * np.identity(self.A.shape[1])

        if self.A is None and self.B is None:
            raise NotFittedError(
                "At least one call of batch_update on data must be done "
                "before solving for weights."
            )
        if self.hyperparameters.use_least_squares_solve:
            W = np.linalg.lstsq(self.A + reg, self.B)[0]
        else:
            W = np.linalg.solve(self.A + reg, self.B)
        coefficients, intercepts = W[:-1, :], W[-1, :]
        return coefficients, intercepts


class ReservoirComputingReadout:
    """Readout layer of the reservoir computing model
    coefficients: linear regression coefficients
    intercepts: linear regression intercepts
    """

    _COEFFICIENTS_NAME = "coefficients.npz"
    _INTERCEPTS_NAME = "intercepts.npy"

    def __init__(self, coefficients: np.ndarray, intercepts: np.ndarray):
        self.coefficients = coefficients
        self.intercepts = intercepts

    def predict(self, input: np.ndarray):
        return (input @ self.coefficients) + self.intercepts

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        # convert to sparse before saving so a single dump/load can be used
        with fsspec.open(os.path.join(path, self._COEFFICIENTS_NAME), "wb") as f:
            scipy.sparse.save_npz(f, scipy.sparse.coo_matrix(self.coefficients))
        with fsspec.open(os.path.join(path, self._INTERCEPTS_NAME), "wb") as f:
            np.save(f, self.intercepts)

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingReadout":
        with fsspec.open(os.path.join(path, cls._COEFFICIENTS_NAME), "rb") as f:
            coefficients = scipy.sparse.load_npz(f)
        with fsspec.open(os.path.join(path, cls._INTERCEPTS_NAME), "rb") as f:
            intercepts = np.load(f)
        return cls(coefficients=coefficients, intercepts=intercepts)


def combine_readouts(readouts: Sequence[ReservoirComputingReadout]):
    coefs, intercepts = [], []
    for readout in readouts:
        coefs.append(readout.coefficients)
        intercepts.append(readout.intercepts)

    # Merge the coefficient arrays of individual readouts into single
    # block diagonal matrix
    combined_coefficients = scipy.sparse.block_diag(coefs)

    # Concatenate the intercepts of individual readouts into single array
    combined_intercepts = np.concatenate(intercepts)

    return ReservoirComputingReadout(
        coefficients=combined_coefficients, intercepts=combined_intercepts,
    )
