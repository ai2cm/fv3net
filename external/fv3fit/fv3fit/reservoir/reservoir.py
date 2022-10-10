import logging
import numpy as np
import scipy

from .config import ReservoirHyperparameters


logger = logging.getLogger(__name__)


class Reservoir:
    def __init__(
        self, hyperparameters: ReservoirHyperparameters,
    ):
        self.hyperparameters = hyperparameters

        self.W_in = self._generate_W_in()
        self.W_res = self._generate_W_res()
        self.state = np.zeros(self.hyperparameters.reservoir_state_dim)

    def increment_state(self, input):
        self.state = np.tanh(self.W_in * input + self.W_res * self.state)

    def reset_state(self):
        logger.info("Resetting reservoir state.")
        self.state = np.zeros(self.hyperparameters.reservoir_state_dim)

    def synchronize(self, synchronization_time_series):
        self.reset_state()
        for input in synchronization_time_series:
            self.increment_reservoir_state(input)

    def _random_uniform_sample_func(self, min=0.0, max=1.0):
        def _f(d):
            return np.random.uniform(min, max, size=d)

        return _f

    def _generate_W_in(self):
        W_in_rows = []
        # Generate by row to ensure same number of connections per input element
        np.random.seed(self.hyperparameters.seed)
        for k in range(self.hyperparameters.reservoir_state_dim):
            W_in_rows.append(
                scipy.sparse.random(
                    m=self.hyperparameters.input_dim,
                    n=1,
                    density=1 - self.hyperparameters.input_coupling_sparsity,
                    data_rvs=self._random_uniform_sample_func(
                        min=-self.hyperparameters.input_coupling_scaling,
                        max=self.hyperparameters.input_coupling_scaling,
                    ),
                )
            )

        return scipy.sparse.hstack(W_in_rows).T

    def _generate_W_res(self):
        np.random.seed(self.hyperparameters.seed)
        W_res = scipy.sparse.random(
            m=self.hyperparameters.reservoir_state_dim,
            n=self.hyperparameters.reservoir_state_dim,
            density=1.0 - self.hyperparameters.sparsity,
            data_rvs=self._random_uniform_sample_func(min=0, max=1.0),
        )
        if not self.hyperparameters.res_scaling:
            scaling = self.hyperparameters.spectral_radius / max(
                abs(scipy.sparse.linalg.eigs(W_res)[0])
            )
        else:
            scaling = self.hyperparameters.res_scaling
        return scaling * W_res
