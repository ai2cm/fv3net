import logging
import numpy as np
import scipy
from typing import Optional

from .config import ReservoirHyperparameters

logger = logging.getLogger(__name__)


def _random_uniform_sample_func(min, max):
    def _f(d):
        return np.random.uniform(min, max, size=d)

    return _f


def _random_uniform_sparse_matrix(m, n, sparsity, min=0, max=1):
    return scipy.sparse.random(
        m=m,
        n=n,
        density=1.0 - sparsity,
        data_rvs=_random_uniform_sample_func(min=min, max=max),
    )


class Reservoir:
    def __init__(
        self,
        hyperparameters: ReservoirHyperparameters,
        input_size: int,
        W_in: Optional[scipy.sparse.coo_matrix] = None,
        W_res: Optional[scipy.sparse.coo_matrix] = None,
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
        self.input_size = input_size

        np.random.seed(self.hyperparameters.seed)
        self.W_in = W_in if W_in is not None else self._generate_W_in()
        self.W_res = W_res if W_res is not None else self._generate_W_res()

    def __getattr__(self, attr):
        if attr == "state":
            raise AttributeError(
                "Reservoir.state is not set yet."
                "To initialize, first use Reservoir.reset_state"
            )
        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute {attr}."
        )

    def increment_state(self, input):
        self.state = np.tanh(self.W_in @ input + self.W_res @ self.state)

    def reset_state(self, input_shape: tuple):
        logger.info("Resetting reservoir state.")
        if len(input_shape) > 1:
            # Input is a 2d matrix with each colum as a separate subdomain
            ncols_inputs = input_shape[1]
            state_after_reset = np.zeros(
                (self.hyperparameters.state_size, ncols_inputs)
            )
        elif len(input_shape) == 1:
            # Input is a 1d vector
            state_after_reset = np.zeros(self.hyperparameters.state_size)
        else:
            raise ValueError("Input shape tuple must describe either a 1D or 2D array.")
        self.state = state_after_reset

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
