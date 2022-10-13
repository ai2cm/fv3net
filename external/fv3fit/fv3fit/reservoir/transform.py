import numpy as np


class InputNoise:
    def __init__(self, size: int, stddev: float, seed: int = 0):
        self.size = size
        self.stddev = stddev
        self.seed = seed
        np.random.seed(self.seed)

    def generate(self):
        return np.random.normal(loc=0, scale=self.stddev, size=self.size)
