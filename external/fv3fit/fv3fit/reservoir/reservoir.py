import logging
import numpy as np
import scipy

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
        self, hyperparameters: ReservoirHyperparameters,
    ):
        self.hyperparameters = hyperparameters
        np.random.seed(self.hyperparameters.seed)
        self.W_in = self._generate_W_in()
        self.W_res = self._generate_W_res()
        self.state_after_reset = (
            np.zeros(
                (self.hyperparameters.state_size, self.hyperparameters.ncols_state)
            )
            if self.hyperparameters.ncols_state > 1
            else np.zeros(self.hyperparameters.state_size)
        )
        self.state = self.state_after_reset

    def increment_state(self, input):
        self.state = np.tanh(self.W_in @ input + self.W_res @ self.state)

    def reset_state(self):
        logger.info("Resetting reservoir state.")
        self.state = self.state_after_reset

    def synchronize(self, synchronization_time_series):
        self.reset_state()
        for input in synchronization_time_series:
            self.increment_state(input)

    def _generate_W_in(self):
        W_in_cols = []
        # Generate by column to ensure same number of connections per input element,
        # as described in Wikner+ 2020 (https://doi.org/10.1063/5.0005541)
        for k in range(self.hyperparameters.input_size):
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
