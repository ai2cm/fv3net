import dacite
import dataclasses
import fsspec
import io
import joblib
import numpy as np
import re
import scipy.sparse
from typing import Sequence, Optional

from .config import ReadoutHyperparameters, BatchLinearRegressorHyperparameters


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
                "Last column of X array must all be ones if add_bias_term " "is False."
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

    hyperparameters: hyperparameters describing the readout
    coefficients: if provided from an already-fit readout,
        use as the linear regression coefficients
    intercepts: if provided from an already-fit readout,
        use as the linear regression intercepts
    """

    _READOUT_NAME = "readout.bin"
    _COEFFICIENTS_NAME = "readout_coefficients.npz"

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
        self.linear_regressor = BatchLinearRegressor(
            hyperparameters.linear_regressor_config
        )

    # TODO: Remove this method from readout and directly call square_even_terms and
    # BatchLinearRegressor.batch_update in the training function. The saved
    # Predictor and its components should not have a fit method.
    def fit(
        self,
        res_states: np.ndarray,
        output_states: np.ndarray,
        calculate_weights: bool = True,
    ) -> None:
        if self.square_half_hidden_state is True:
            res_states = square_even_terms(res_states, axis=1)
        self.linear_regressor.batch_update(res_states, output_states)
        if calculate_weights:
            self.calculate_weights()

    def calculate_weights(self) -> None:
        self.coefficients, self.intercepts = self.linear_regressor.get_weights()

    def predict(self, input: np.ndarray):
        if self.square_half_hidden_state:
            input = square_even_terms(input, axis=0)
        if len(input.shape) > 1:
            print(f"input shape before flattening: {input.shape}")
            input = input.reshape(-1)
            print(f"input shape after flattening: {input.shape}")

        flat_prediction = (input * self.coefficients) + self.intercepts
        print(f"flat_prediction shape: {flat_prediction.shape}")

        if len(input.shape) > 1:
            return flat_prediction.reshape(-1, input.shape[-1])
        else:
            return flat_prediction

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)
        mapper = fs.get_mapper(path)
        components = {
            "lr_hyperparameters": dataclasses.asdict(
                self.hyperparameters.linear_regressor_config
            ),
            "square_half_hidden_state": self.hyperparameters.square_half_hidden_state,
            # "coefficients": self.coefficients,
            "intercepts": self.intercepts,
        }
        f = io.BytesIO()
        joblib.dump(components, f)
        mapper[self._READOUT_NAME] = f.getvalue()
        with fs.open(f"{path}/{self._COEFFICIENTS_NAME}", "wb") as f:
            scipy.sparse.save_npz(f, self.coefficients)

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingReadout":
        mapper = fsspec.get_mapper(path)
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        f = io.BytesIO(mapper[cls._READOUT_NAME])
        readout_components = joblib.load(f)
        lr_config = dacite.from_dict(
            BatchLinearRegressorHyperparameters,
            readout_components.pop("lr_hyperparameters"),
        )
        hyperparameters = ReadoutHyperparameters(
            linear_regressor_config=lr_config,
            square_half_hidden_state=readout_components.pop("square_half_hidden_state"),
        )
        with fs.open(f"{path}/{cls._COEFFICIENTS_NAME}", "rb") as f:
            coefficients = scipy.sparse.load_npz(f)
        return cls(
            hyperparameters=hyperparameters,
            coefficients=coefficients,
            **readout_components,
        )


def combine_readouts(readouts: Sequence[ReservoirComputingReadout]):
    intercepts, square_state_settings, lr_config_hashes = [], [], []
    for readout in readouts:
        # coefs.append(readout.coefficients)
        intercepts.append(readout.intercepts)
        square_state_settings.append(readout.square_half_hidden_state)
        lr_config_hashes.append(readout.hyperparameters.linear_regressor_config.hash)

    if len(np.unique(square_state_settings)) != 1:
        raise ValueError(
            "All readouts must have the same setting for square_half_hidden_state."
        )
    if len(np.unique(lr_config_hashes)) != 1:
        raise ValueError(
            "All readouts must have the same hyperparameters for BatchLinearRegressor."
        )

    # Merge the coefficient arrays of individual readouts into single
    # block diagonal matrix
    combined_coefficients = scipy.sparse.block_diag(
        [readout.coefficients for readout in readouts]
    )  # scipy.sparse.block_diag(coefs)

    # Concatenate the intercepts of individual readouts into single array
    combined_intercepts = np.concatenate(intercepts)

    hyperparameters = readouts[0].hyperparameters
    return ReservoirComputingReadout(
        hyperparameters=hyperparameters,
        coefficients=combined_coefficients,
        intercepts=combined_intercepts,
    )


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
        return input * self.coefficients + self.intercepts

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
            lr_config = dacite.from_dict(
                BatchLinearRegressorHyperparameters,
                readout_components.pop("lr_hyperparameters"),
            )
            hyperparameters = ReadoutHyperparameters(
                linear_regressor_config=lr_config,
                square_half_hidden_state=readout_components.pop(
                    "square_half_hidden_state"
                ),
            )
            readouts.append(
                ReservoirComputingReadout(
                    hyperparameters=hyperparameters, **readout_components
                )
            )

        return cls(readouts)
