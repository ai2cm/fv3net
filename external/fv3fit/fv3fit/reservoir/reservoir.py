import dataclasses
import logging
import numpy as np
import scipy
from typing import Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReservoirHyperparameters:
    """Hyperparameters for reservoir

    input_dim: Size of input vector
    reservoir_state_dim: Size of hidden state vector,
        W_res has shape reservoir_state_dim x reservoir_state_dim
    sparsity: Fraction of elements in W_res that are zero
    output_dim: Size of output vector. Can be smaller than input
        dimension if predicting on subdomains with overlapping
        input regions.
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
    res_scaling: Optional scale value for W_res that can be provided in lieu of
        spectral radius. This is useful if you know what scaling parameter
        applied to the uniform distribution [0, 1] for a given reservoir size
        will lead to the (approximate) desired spectral radius, since eigenvalue
        calculation for larger reservoirs can be slow.
    """

    input_dim: int
    reservoir_state_dim: int
    sparsity: float
    output_dim: Optional[int] = None

    spectral_radius: Optional[float] = None

    seed: int = 0
    input_coupling_sparsity: float = 0.0
    input_coupling_scaling: float = 1.0
    res_scaling: Optional[float] = None

    def __post_init__(self):
        if self.spectral_radius and self.scaling:
            raise ValueError("Only one of spectral_radius or scaling can be specified")
        if not self.output_dim:
            self.output_dim = self.input_dim


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
