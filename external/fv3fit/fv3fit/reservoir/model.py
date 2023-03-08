import fsspec
import numpy as np
import os
import scipy.sparse
from typing import Optional, Union
import yaml

from .reservoir import Reservoir
from .domain import RankDivider


def _square_evens(v: np.ndarray) -> np.ndarray:
    evens = v[::2]
    odds = v[1::2]
    c = np.empty((v.size,), dtype=v.dtype)
    c[0::2] = evens ** 2
    c[1::2] = odds
    return c


def square_even_terms(v: np.ndarray, axis=1) -> np.ndarray:
    return np.apply_along_axis(func1d=_square_evens, axis=axis, arr=v)


class ReservoirComputingModel:
    _RESERVOIR_SUBDIR = "reservoir"
    _RANK_DIVIDER_NAME = "rank_divider"
    _COEFFICIENTS_NAME = "coefficients.npz"
    _INTERCEPTS_NAME = "intercepts.npz"
    _METADATA_NAME = "metadata.yaml"

    def __init__(
        self,
        reservoir: Reservoir,
        coefficients: Union[np.ndarray, scipy.sparse.coo_matrix],
        intercepts: Union[np.ndarray, scipy.sparse.coo_matrix],
        square_half_hidden_state: bool = False,
        rank_divider: Optional[RankDivider] = None,
    ):
        """_summary_

        Args:
            reservoir: Reservoir which takes input and updates hidden state
            coefficients: _description_
            intercepts:
            square_half_hidden_state: if True, square even terms in the reservoir
                state before it is used as input to the regressor's .fit and
                .predict methods. This option was found to be important for skillful
                predictions in Wikner+2020 (https://doi.org/10.1063/5.0005541).
            rank_divider: If used in training, save the rank divider that converts
                the full 3D rank into stacked subdomains.
        """
        self.reservoir = reservoir
        self.coefficients = coefficients
        self.intercepts = intercepts
        self.square_half_hidden_state = square_half_hidden_state
        self.rank_divider = rank_divider

    def predict(self):
        if self.square_half_hidden_state is True:
            input = square_even_terms(self.reservoir.state, axis=0)
        else:
            input = self.reservoir.state
        prediction = input * self.coefficients + self.intercepts
        self.reservoir.increment_state(prediction)
        return prediction

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        if self.rank_divider is not None:
            self.rank_divider.dump(os.path.join(path, self._RANK_DIVIDER_NAME))
        self.reservoir.dump(f"{path}/{self._RESERVOIR_SUBDIR}")

        # Save and load as sparse COO matrices so a common dump/load can be used
        with fsspec.open(os.path.join(path, self._COEFFICIENTS_NAME), "wb") as f:
            scipy.sparse.save_npz(f, scipy.sparse.coo_matrix(self.coefficients))
        with fsspec.open(os.path.join(path, self._INTERCEPTS_NAME), "wb") as f:
            scipy.sparse.save_npz(f, scipy.sparse.coo_matrix(self.intercepts))

        metadata = {"square_half_hidden_state": self.square_half_hidden_state}
        with fsspec.open(os.path.join(path, self._METADATA_NAME), "w") as f:
            f.write(yaml.dump(metadata))


'''
    @classmethod
    def load(cls, path: str) -> "ReservoirComputingModel":
        """Load a model from a remote path"""
        reservoir = Reservoir.load(f"{path}/{cls._RESERVOIR_SUBDIR}")
        return
'''
