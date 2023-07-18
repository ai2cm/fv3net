import fsspec
import numpy as np
import os
import scipy.sparse
from typing import Optional, Sequence, cast

from .config import BatchLinearRegressorHyperparameters


class NotFittedError(Exception):
    def __init__(self, message):
        super().__init__(message)


class BatchLinearRegressor:
    """ Solves for weights W in matrix equation AW=B
    Where A = X_T.X + l2*I and B= X_T.y
    """

    def __init__(self, hyperparameters: BatchLinearRegressorHyperparameters):
        self.hyperparameters = hyperparameters
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None

    def _add_bias_feature(self, X):
        leading_shape = list(X.shape[:-1]) + [1]
        return np.concatenate([X, np.ones(leading_shape)], axis=1)

    def _check_X_last_col_constant(self, X):
        last_col = X[..., -1]
        if not np.allclose(np.unique(last_col), np.array([1.0])):
            raise ValueError(
                "Last column of X array must all be ones if add_bias_term is False."
            )

    def batch_update(self, X: np.ndarray, y: np.ndarray):
        if self.hyperparameters.add_bias_term:
            X = self._add_bias_feature(X)
        else:
            self._check_X_last_col_constant(X)

        A_update = np.dot(X.T, X)
        B_update = np.dot(X.T, y)

        if self.A is None and self.B is None:
            self.A = A_update
            self.B = B_update
        else:
            self.A = np.add(cast(np.ndarray, self.A), A_update)
            self.B = np.add(cast(np.ndarray, self.B), B_update)

    def get_weights(self):
        # use_least_squares_solve is useful for simple test cases
        # where np.linalg.solve encounters for singular XT.X

        reg = self.hyperparameters.l2 * np.identity(self.A.shape[-1])

        if self.A is None and self.B is None:
            raise NotFittedError(
                "At least one call of batch_update on data must be done "
                "before solving for weights."
            )
        if self.hyperparameters.use_least_squares_solve:
            # TODO: not sure if lstsq works with more than 2D?
            W = np.linalg.lstsq(cast(np.ndarray, self.A) + reg, self.B)[0]
        else:
            W = np.linalg.solve(cast(np.ndarray, self.A) + reg, self.B)
        # TODO: remove elipsis if creating a more specific readout
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
