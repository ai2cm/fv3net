import numpy as np


class InputNoise:
    def __init__(self, dim: int, stddev: float, seed: int = 0):
        self.dim = dim
        self.stddev = stddev
        self.seed = seed
        np.random.seed(self.seed)

    def generate(self):
        return np.random.normal(loc=0, scale=self.stddev, size=self.dim)


def square_half_in_place(self, v: np.ndarray) -> np.ndarray:
    evens = v[::2]
    odds = v[1::2]
    c = np.empty((v.size,), dtype=v.dtype)
    c[0::2] = evens ** 2
    c[1::2] = odds
    return c


def concat_quadratic_terms(self, v: np.ndarray) -> np.ndarray:
    return np.hstack([v, v ** 2])
