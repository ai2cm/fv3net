import dataclasses
import logging
import numpy as np
import scipy
from typing import Optional

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
        data_rvs=_random_uniform_sample_func(min=0, max=1.0),
    )


@dataclasses.dataclass
class ReservoirHyperparameters:
    """Hyperparameters for reservoir

    input_size: Size of input vector
    state_size: Size of hidden state vector,
        W_res has shape state_size x state_size
    adjacency_matrix_sparsity: Fraction of elements in adjacency matrix
        W_res that are zero
    output_size: Optional: size of output vector. Can be smaller than input
        dimension if predicting on subdomains with overlapping
        input regions. Defaults to same as input_size
    spectral_radius: Largest absolute value eigenvalue of W_res.
        Larger values increase the memory of the reservoir.
    seed: Random seed for sampling
    input_coupling_sparsity: Fraction of elements in each row of W_in
        that are zero. Kept the same in all rows to ensure each input
        is equally connected into the reservoir. Defaults to 0.
    input_coupling_scaling: Scaling applied to W_in. Defaults to 1,
        where all elements are sampled from random uniform distribution
        [-1, 1]. Changing this affects relative weighting of reservoir memory
        versus the most recent state.
    """

    input_size: int
    state_size: int
    adjacency_matrix_sparsity: float
    spectral_radius: float
    output_size: Optional[int] = None
    seed: int = 0
    input_coupling_sparsity: float = 0.0
    input_coupling_scaling: float = 1.0

    def __post_init__(self):
        if not self.output_size:
            self.output_size = self.input_size


class Reservoir:
    def __init__(
        self, hyperparameters: ReservoirHyperparameters,
    ):
        self.hyperparameters = hyperparameters
        np.random.seed(self.hyperparameters.seed)
        self.W_in = self._generate_W_in()
        self.W_res = self._generate_W_res()
        self.state = np.zeros(self.hyperparameters.state_size)

    def increment_state(self, input):
        self.state = np.tanh(self.W_in * input + self.W_res * self.state)

    def reset_state(self):
        logger.info("Resetting reservoir state.")
        self.state = np.zeros(self.hyperparameters.state_size)

    def synchronize(self, synchronization_time_series):
        self.reset_state()
        for input in synchronization_time_series:
            self.increment_reservoir_state(input)

    def _generate_W_in(self):
        W_in_rows = []
        # Generate by row to ensure same number of connections per input element
        for k in range(self.hyperparameters.state_size):
            W_in_rows.append(
                _random_uniform_sparse_matrix(
                    m=1,
                    n=self.hyperparameters.input_size,
                    sparsity=self.hyperparameters.input_coupling_sparsity,
                    min=-self.hyperparameters.input_coupling_scaling,
                    max=self.hyperparameters.input_coupling_scaling,
                )
            )

        return scipy.sparse.vstack(W_in_rows)

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
