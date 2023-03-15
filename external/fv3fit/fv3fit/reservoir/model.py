import fsspec
from fv3fit.reservoir.readout import ReservoirComputingReadout
import numpy as np
import os
from typing import Optional, Iterable, Hashable
import yaml

from fv3fit import Predictor
from .._shared import register_training_function, StandardScaler
from .reservoir import Reservoir
from .domain import RankDivider
from .config import ReservoirTrainingConfig


def _square_evens(v: np.ndarray) -> np.ndarray:
    evens = v[::2]
    odds = v[1::2]
    c = np.empty((v.size,), dtype=v.dtype)
    c[0::2] = evens ** 2
    c[1::2] = odds
    return c


def _square_even_terms(v: np.ndarray, axis=1) -> np.ndarray:
    return np.apply_along_axis(func1d=_square_evens, axis=axis, arr=v)


def _exists_in_dir(file_name, dir):
    fs = fsspec.get_fs_token_paths(dir)[0]
    contents = [os.path.basename(os.path.normpath(p)) for p in fs.ls(dir)]
    return file_name in contents


@register_training_function("pure-reservoir", ReservoirTrainingConfig)
class ReservoirComputingModel(Predictor):
    _RESERVOIR_SUBDIR = "reservoir"
    _READOUT_SUBDIR = "readout"
    _METADATA_NAME = "metadata.yaml"
    _SCALER_NAME = "scaler.npz"
    _RANK_DIVIDER_NAME = "rank_divider.yaml"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        reservoir: Reservoir,
        readout: ReservoirComputingReadout,
        square_half_hidden_state: bool = False,
        rank_divider: Optional[RankDivider] = None,
        scaler: Optional[StandardScaler] = None,
    ):
        """_summary_

        Args:
            reservoir: Reservoir which takes input and updates hidden state
            readout: readout layer which takes in state and predicts next time step
            square_half_hidden_state: if True, square even terms in the reservoir
                state before it is used as input to the regressor's .fit and
                .predict methods. This option was found to be important for skillful
                predictions in Wikner+2020 (https://doi.org/10.1063/5.0005541).
            rank_divider: object used to divide and reconstruct domain <-> subdomains
        """
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.reservoir = reservoir
        self.readout = readout
        self.square_half_hidden_state = square_half_hidden_state
        self.scaler = scaler
        self.rank_divider = rank_divider

    def predict(self):
        if self.square_half_hidden_state is True:
            readout_input = _square_even_terms(self.reservoir.state, axis=0)
        else:
            readout_input = self.reservoir.state
        # For prediction over multiple subdomains (>1 column in reservoir state
        # array), flatten state into 1D vector before predicting
        if len(readout_input.shape) > 1:
            readout_input = readout_input.reshape(-1)
        prediction = self.readout.predict(readout_input).reshape(-1)

        return prediction

    def increment_state(self, prediction_with_overlap):
        self.reservoir.increment_state(prediction_with_overlap)

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        self.reservoir.dump(os.path.join(path, self._RESERVOIR_SUBDIR))
        self.readout.dump(os.path.join(path, self._READOUT_SUBDIR))

        metadata = {
            "square_half_hidden_state": self.square_half_hidden_state,
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
        }
        with fsspec.open(os.path.join(path, self._METADATA_NAME), "w") as f:
            f.write(yaml.dump(metadata))

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        if self.scaler is not None:
            with fs.open(f"{path}/{self._SCALER_NAME}", "wb") as f:
                self.scaler.dump(f)
        if self.rank_divider is not None:
            self.rank_divider.dump(os.path.join(path, self._RANK_DIVIDER_NAME))

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingModel":
        """Load a model from a remote path"""
        reservoir = Reservoir.load(os.path.join(path, cls._RESERVOIR_SUBDIR))
        readout = ReservoirComputingReadout.load(
            os.path.join(path, cls._READOUT_SUBDIR)
        )
        with fsspec.open(os.path.join(path, cls._METADATA_NAME), "r") as f:
            metadata = yaml.safe_load(f)

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        if _exists_in_dir(cls._SCALER_NAME, path):
            with fs.open(f"{path}/{cls._SCALER_NAME}", "rb") as f:
                scaler = StandardScaler.load(f)
        else:
            scaler = None
        if _exists_in_dir(cls._RANK_DIVIDER_NAME, path):
            rank_divider = RankDivider.load(os.path.join(path, cls._RANK_DIVIDER_NAME))
        else:
            rank_divider = None

        return cls(
            input_variables=metadata["input_variables"],
            output_variables=metadata["output_variables"],
            reservoir=reservoir,
            readout=readout,
            square_half_hidden_state=metadata["square_half_hidden_state"],
            scaler=scaler,
            rank_divider=rank_divider,
        )
