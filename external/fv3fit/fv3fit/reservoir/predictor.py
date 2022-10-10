import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from typing import Union
from .reservoir import Reservoir


def _square_even_terms(v: np.ndarray) -> np.ndarray:
    evens = v[::2]
    odds = v[1::2]
    c = np.empty((v.size,), dtype=v.dtype)
    c[0::2] = evens ** 2
    c[1::2] = odds
    return c


class ReservoirPredictor:
    def __init__(
        self,
        reservoir: Reservoir,
        linreg: Union[Ridge, LinearRegression],
        quadratic_even_terms: bool = False,
    ):
        self.reservoir = reservoir
        self.linreg = linreg
        self.quadratic_even_terms = quadratic_even_terms

    def predict(self):
        # the reservoir state at t+Delta t uses the state AND input at t,
        # so the prediction occurs before the state increment
        res_state_ = self.reservoir.state.reshape(1, -1)
        if self.quadratic_even_terms:
            res_state_ = _square_even_terms(res_state_)

        prediction = self.linreg.predict(res_state_).reshape(-1)
        self.reservoir.increment_state(prediction)
        return prediction
