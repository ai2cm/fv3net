import numpy as np


class InputNoise:
    def __init__(self, dim: int, stddev: float, seed: int = 0):
        self.dim = dim
        self.stddev = stddev
        self.seed = seed
        np.random.seed(self.seed)

    def generate(self):
        return np.random.normal(loc=0, scale=self.stddev, size=self.dim)
