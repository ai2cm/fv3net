import dacite
import dataclasses
import fsspec
import logging
import numpy as np
import os
import scipy
from typing import cast, Optional
import yaml

from .config import ReservoirHyperparameters

logger = logging.getLogger(__name__)


def _random_uniform_sample_func(min, max):
    def _f(d):
        return np.random.uniform(min, max, size=d)

    return _f


def _random_uniform_sparse_matrix(m, n, sparsity, min=0, max=1, type="csc"):
    return scipy.sparse.random(
        m=m,
        n=n,
        density=1.0 - sparsity,
        data_rvs=_random_uniform_sample_func(min=min, max=max),
        format=type,
    )


class Reservoir:
    _INPUT_WEIGHTS_NAME = "reservoir_W_in.npz"
    _RESERVOIR_WEIGHTS_NAME = "reservoir_W_res.npz"
    _METADATA_NAME = "metadata.bin"
    _INPUT_MASK_NAME = "input_mask.npy"
    _STATE_NAME = "state.npy"

    def __init__(
        self,
        hyperparameters: ReservoirHyperparameters,
        input_size: int,
        W_in: Optional[scipy.sparse.csc_matrix] = None,
        W_res: Optional[scipy.sparse.csc_matrix] = None,
        input_mask_array: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
    ):
        """

        Args:
            hyperparameters: information for generating reservoir matrices
            input_size: length of input vector features
            W_in: Weights for input matrix. If None, this matrix will be generated
                upon initialization.
            W_res: Weights for reservoir matrix. If None, this matrix will be
                generated upon initialiation.
        """
        self.hyperparameters = hyperparameters
        self.input_size = int(input_size)

        np.random.seed(self.hyperparameters.seed)
        self.W_in = W_in if W_in is not None else self._generate_W_in()
        self.W_res = W_res if W_res is not None else self._generate_W_res()
        self.state = state
        self.input_mask_array = input_mask_array

    def increment_state(self, input):
        # input: [subdomain, features]
        # (optional) input_mask: [subdomain, features]
        # W_in: [features, state_size]
        # W_res: [state_size, state_size]
        # output: [subdomain, state_size]
        if self.input_mask_array is not None:
            masked_input = input * self.input_mask_array
        else:
            masked_input = input
        self.state: np.ndarray = np.tanh(
            masked_input @ self.W_in.T + self.state @ self.W_res.T
        )

    def reset_state(self, input_shape: tuple):
        logger.info("Resetting reservoir state.")
        if len(input_shape) > 1:
            # Input is a 2d matrix with each row as a separate subdomain
            input_subdomains = input_shape[0]
            state_after_reset = np.zeros(
                (input_subdomains, self.hyperparameters.state_size)
            )
        elif len(input_shape) == 1:
            # Input is a 1d vector
            state_after_reset = np.zeros(self.hyperparameters.state_size)
        else:
            raise ValueError("Input shape tuple must describe either a 1D or 2D array.")
        self.state = state_after_reset

    def set_state(self, new_state: np.ndarray):
        if self.state is not None:
            if self.state.shape != new_state.shape:
                raise ValueError("Provided state does not match reservoir state shape")
        self.state = new_state

    def synchronize(self, synchronization_time_series):
        self.reset_state(input_shape=synchronization_time_series[0].shape)
        for input in synchronization_time_series:
            self.increment_state(input)

    def _generate_W_in(self):
        W_in_cols = []
        # Generate by column to ensure same number of connections per input element,
        # as described in Wikner+ 2020 (https://doi.org/10.1063/5.0005541)
        for k in range(self.input_size):
            W_in_cols.append(
                _random_uniform_sparse_matrix(
                    m=self.hyperparameters.state_size,
                    n=1,
                    sparsity=self.hyperparameters.input_coupling_sparsity,
                    min=-self.hyperparameters.input_coupling_scaling,
                    max=self.hyperparameters.input_coupling_scaling,
                )
            )

        return scipy.sparse.hstack(W_in_cols)

    def _generate_W_res(self):
        W_res = _random_uniform_sparse_matrix(
            m=self.hyperparameters.state_size,
            n=self.hyperparameters.state_size,
            sparsity=self.hyperparameters.adjacency_matrix_sparsity,
            min=0,
            max=1,
        )
        largest_magnitude_eigval = scipy.sparse.linalg.eigs(
            W_res, return_eigenvectors=False, k=1, which="LM"
        ).item()
        scaling = self.hyperparameters.spectral_radius / abs(largest_magnitude_eigval)

        return scaling * W_res

    def dump_state(self, path: str) -> None:
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)

        if self.state is not None:
            with fs.open(os.path.join(path, self._STATE_NAME), "wb") as f:
                np.save(f, self.state)

    def dump(self, path: str) -> None:
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)
        mapper = fs.get_mapper(path)
        metadata = {
            "reservoir_hyperparameters": dataclasses.asdict(self.hyperparameters,),
            "input_size": self.input_size,
        }
        with fs.open(os.path.join(path, self._INPUT_WEIGHTS_NAME), "wb") as f:
            scipy.sparse.save_npz(f, self.W_in)
        with fs.open(os.path.join(path, self._RESERVOIR_WEIGHTS_NAME), "wb") as f:
            scipy.sparse.save_npz(f, self.W_res)

        if self.input_mask_array is not None:
            with fsspec.open(os.path.join(path, self._INPUT_MASK_NAME), "wb") as f:
                np.save(f, self.input_mask_array, allow_pickle=False)

        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")

    @classmethod
    def load(cls, path: str) -> "Reservoir":
        mapper = fsspec.get_mapper(path)
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]

        metadata = yaml.safe_load(mapper[cls._METADATA_NAME])

        reservoir_hyperparameters = dacite.from_dict(
            ReservoirHyperparameters, metadata["reservoir_hyperparameters"]
        )
        with fs.open(f"{path}/{cls._INPUT_WEIGHTS_NAME}", "rb") as f:
            reservoir_W_in = scipy.sparse.load_npz(f)
        with fs.open(f"{path}/{cls._RESERVOIR_WEIGHTS_NAME}", "rb") as f:
            reservoir_W_res = scipy.sparse.load_npz(f)
        try:
            with fsspec.open(os.path.join(path, cls._INPUT_MASK_NAME), "rb") as f:
                input_mask_array: Optional[np.ndarray] = np.load(f)
        except (FileNotFoundError):
            input_mask_array = None

        try:
            with fsspec.open(os.path.join(path, cls._STATE_NAME), "rb") as f:
                state = cast(np.ndarray, np.load(f))
        except (FileNotFoundError):
            state = None

        return cls(
            reservoir_hyperparameters,
            W_in=reservoir_W_in,
            W_res=reservoir_W_res,
            input_size=metadata["input_size"],
            input_mask_array=input_mask_array,
            state=state,
        )
