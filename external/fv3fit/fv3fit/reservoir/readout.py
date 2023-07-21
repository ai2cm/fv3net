import fsspec
import numpy as np
import os
from typing import Optional, Sequence

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
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    def _check_X_last_col_constant(self, X):
        last_col = X[:, -1]
        if not np.allclose(np.unique(last_col), np.array([1.0])):
            raise ValueError(
                "Last column of X array must all be ones if add_bias_term is False."
            )

    def batch_update(self, X: np.ndarray, y: np.ndarray):
        # X is [time, feature]
        # y is [time, feature]
        if self.hyperparameters.add_bias_term:
            X = self._add_bias_feature(X)
        else:
            self._check_X_last_col_constant(X)

        if self.A is None and self.B is None:
            self.A = np.zeros((X.shape[1], X.shape[1]))
            self.B = np.zeros((X.shape[1], y.shape[1]))

        self.A = np.add(self.A, np.dot(X.T, X))
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
        # coefficients: [(subdomain), in_features, out_features]
        # intercepts: [(subdomain), out_features]
        self.coefficients = coefficients
        self.intercepts = intercepts

        if len(coefficients.shape) == 2:
            self._einsum_str = "...j,jk->...k"
        elif len(coefficients.shape) == 3:
            self._einsum_str = "...ij,ijk->...ik"
        else:
            raise ValueError(
                "Coefficients must be a 2D or 3D array. "
                f"Got coefficients with shape {coefficients.shape}"
            )

    def predict(self, input: np.ndarray):
        return np.einsum(self._einsum_str, input, self.coefficients) + self.intercepts

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        # convert to sparse before saving so a single dump/load can be used
        with fsspec.open(os.path.join(path, self._COEFFICIENTS_NAME), "wb") as f:
            np.save(f, self.coefficients, allow_pickle=False)
        with fsspec.open(os.path.join(path, self._INTERCEPTS_NAME), "wb") as f:
            np.save(f, self.intercepts, allow_pickle=False)

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingReadout":
        with fsspec.open(os.path.join(path, cls._COEFFICIENTS_NAME), "rb") as f:
            coefficients = np.load(f)
        with fsspec.open(os.path.join(path, cls._INTERCEPTS_NAME), "rb") as f:
            intercepts = np.load(f)
        return cls(coefficients=coefficients, intercepts=intercepts)


def combine_readouts_from_subdomain_regressors(
    regressors: Sequence[BatchLinearRegressor],
):
    all_coefficients, all_intercepts = [], []
    for r in regressors:
        coefs_, intercept = r.get_weights()
        all_coefficients.append(coefs_)
        all_intercepts.append(intercept)

    # Concatenate the intercepts of individual readouts into single array
    combined_coefficients = np.stack(all_coefficients, axis=0)
    combined_intercepts = np.stack(all_intercepts, axis=0)

    return ReservoirComputingReadout(
        coefficients=combined_coefficients, intercepts=combined_intercepts,
    )
