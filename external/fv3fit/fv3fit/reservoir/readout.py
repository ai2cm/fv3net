import io
import joblib
import numpy as np
from sklearn.linear_model import Ridge


def _square_evens(v: np.ndarray) -> np.ndarray:
    evens = v[::2]
    odds = v[1::2]
    c = np.empty((v.size,), dtype=v.dtype)
    c[0::2] = evens ** 2
    c[1::2] = odds
    return c


def square_even_terms(v: np.ndarray, axis=1) -> np.ndarray:
    return np.apply_along_axis(func1d=_square_evens, axis=axis, arr=v)


class ReservoirComputingReadout:
    """Readout layer of the reservoir computing model

    linear_regressor: a sklearn Ridge regreesor
    square_half_hidden_state: if True, square even terms in the reservoir state
        before it is used as input to the regressor's .fit and .predict methods
        This option was found to be important for skillful predictions in
        Wikner+2020 (https://doi.org/10.1063/5.0005541)
    """

    def __init__(
        self, linear_regressor: Ridge, square_half_hidden_state: bool = False,
    ):
        self.linear_regressor = linear_regressor
        self.square_half_hidden_state = square_half_hidden_state

    def fit(self, res_states: np.ndarray, output_states: np.ndarray) -> None:
        if self.square_half_hidden_state:
            res_states = square_even_terms(res_states, axis=1)
        self.linear_regressor.fit(res_states, output_states)

    def predict(self, input):
        if len(input.shape) == 1:
            input = input.reshape(1, -1)
        if self.square_half_hidden_state:
            input = square_even_terms(input, axis=1)
        return self.linear_regressor.predict(input)

    def dumps(self) -> bytes:
        components = {
            "linear_regressor": self.linear_regressor,
            "square_half_hidden_state": self.square_half_hidden_state,
        }
        f = io.BytesIO()
        joblib.dump(components, f)
        return f.getvalue()
