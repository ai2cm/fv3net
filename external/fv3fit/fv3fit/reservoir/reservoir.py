import dataclasses
import logging
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression, Ridge
from typing import Union, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainConfig:
    n_burn: int
    n_samples: None
    noise: float
    l2: float


@dataclasses.dataclass
class ReservoirHyperparameters:
    input_dim: int
    reservoir_state_dim: int
    sparsity: float
    output_dim: Optional[int] = None

    spectral_radius: Optional[float] = None
    square_half: bool = False
    seed: int = 0
    input_coupling_sparsity: float = 0.0
    input_coupling_scaling: float = 1.0
    scaling: Optional[float] = None

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
        self.W_out = None
        self.state = np.zeros(self.hyperparameters.reservoir_state_dim)

    def _random_uniform_sample_func(self, a=1.0):
        def _f(d):
            return np.random.uniform(-a, a, size=d)

        return _f

    def _generate_W_in(self):
        W_in_rows = []
        # Generate by row to ensure same number of nonzero elements per column
        np.random.seed(self.hyperparameters.seed)
        for k in range(self.hyperparameters.input_dim):
            W_in_rows.append(
                scipy.sparse.random(
                    m=self.hyperparameters.reservoir_state_dim,
                    n=1,
                    density=1 - self.hyperparameters.input_coupling_sparsity,
                    data_rvs=self._random_uniform_sample_func(
                        a=self.hyperparameters.input_coupling_scaling
                    ),
                )
            )
        return scipy.sparse.hstack(W_in_rows)

    def _generate_W_res(self):
        np.random.seed(self.hyperparameters.seed)
        W_res = scipy.sparse.random(
            m=self.hyperparameters.reservoir_state_dim,
            n=self.hyperparameters.reservoir_state_dim,
            density=1.0 - self.hyperparameters.sparsity,
            data_rvs=self._random_uniform_sample_func(),
        )
        if not self.hyperparameters.scaling:
            scaling = self.hyperparameters.spectral_radius / max(
                abs(scipy.sparse.linalg.eigs(W_res)[0])
            )
        else:
            scaling = self.hyperparameters.scaling
        return scaling * W_res

    def increment_state(self, input):
        new_state = np.tanh(self.W_in * input + self.W_res * self.state)
        if self.hyperparameters.square_half:
            new_state = self.square_half_state(new_state)
        self.state = new_state

    def square_half_state(self, v):
        evens = v[::2]
        odds = v[1::2]
        c = np.empty((v.size,), dtype=v.dtype)
        c[0::2] = evens ** 2
        c[1::2] = odds
        return c

    def reset_state(self):
        logger.info("Resetting reservoir state.")
        self.state = np.zeros(self.hyperparameters.reservoir_state_dim)

    def synchronize(self, synchronization_time_series):
        self.reset_reservoir_state()
        for input in synchronization_time_series:
            self.increment_reservoir_state(input)


class InputNoise:
    def __init__(self, dim: int, stddev: float, seed: int = 0):
        self.dim = dim
        self.stddev = stddev
        self.seed = seed
        np.random.seed(self.seed)

    def generate(self):
        return np.random.normal(loc=0, scale=self.stddev, size=self.dim)


def transform_inputs_to_reservoir_states(X, reservoir, input_noise: InputNoise):
    reservoir_states = []
    X_noised = X + input_noise.generate()
    for x in X_noised:
        reservoir_states.append(reservoir.state)
        reservoir.increment_state(x)
    return np.array(reservoir_states)


class ReservoirPredictor:
    def __init__(
        self,
        reservoir: Reservoir,
        linreg: Union[Ridge, LinearRegression],
        quadratic_x: bool = False,
    ):
        self.reservoir = reservoir
        self.linreg = linreg
        self.quadratic_x = quadratic_x

    def predict(self, input):
        # the reservoir state at t+Delta t uses the state AND input at t,
        # so the prediction occurs before the state increment
        res_state_ = self.reservoir.state.reshape(1, -1)
        if self.quadratic_x is True:
            prediction_input = np.hstack([res_state_, res_state_ ** 2])
        else:
            prediction_input = res_state_

        prediction = self.linreg.predict(prediction_input)
        self.reservoir.increment_state(input)
        return prediction

    def train_linreg(self, X, y):
        self.linreg.fit(X, y)

    def reset_reservoir_state(self):
        self.reservoir.reset_state()

    def increment_reservoir_state(self, input):
        self.reservoir.increment_state(input)
