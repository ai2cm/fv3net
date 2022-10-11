import dacite
import dataclasses
import fsspec
import io
import joblib
import numpy as np
from sklearn.linear_model import Ridge
import yaml


from .reservoir import Reservoir
from fv3fit.reservoir.config import ReservoirHyperparameters


def _square_even_terms(v: np.ndarray) -> np.ndarray:
    evens = v[::2]
    odds = v[1::2]
    c = np.empty((v.size,), dtype=v.dtype)
    c[0::2] = evens ** 2
    c[1::2] = odds
    return c


class ReservoirPredictor:
    _READOUT_NAME = "readout.pkl"
    _METADATA_NAME = "metadata.bin"

    def __init__(
        self,
        reservoir: Reservoir,
        readout: Ridge,
        square_half_hidden_state: bool = False,
    ):
        self.reservoir = reservoir
        self.readout = readout
        self.square_half_hidden_state = square_half_hidden_state

    def predict(self):
        # the reservoir state at t+Delta t uses the state AND input at t,
        # so the prediction occurs before the state increment
        res_state_ = self.reservoir.state.reshape(1, -1)
        if self.square_half_hidden_state:
            res_state_ = _square_even_terms(res_state_)

        prediction = self.readout.predict(res_state_).reshape(-1)
        self.reservoir.increment_state(prediction)
        return prediction

    def _dumps(self, obj) -> bytes:
        f = io.BytesIO()
        joblib.dump(obj, f)
        return f.getvalue()

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)
        mapper = fs.get_mapper(path)

        mapper[self._READOUT_NAME] = self._dumps(self.readout)
        metadata = {
            "reservoir_hyperparameters": dataclasses.asdict(
                self.reservoir.hyperparameters
            ),
            "square_half_hidden_state": self.square_half_hidden_state,
        }
        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")

    @classmethod
    def load(cls, path: str) -> "ReservoirPredictor":
        """Load a model from a remote path"""
        mapper = fsspec.get_mapper(path)

        f = io.BytesIO(mapper[cls._READOUT_NAME])
        readout = joblib.load(f)
        metadata = yaml.safe_load(mapper[cls._METADATA_NAME])

        reservoir_hyperparameters = dacite.from_dict(
            ReservoirHyperparameters, metadata["reservoir_hyperparameters"]
        )

        return cls(
            reservoir=Reservoir(reservoir_hyperparameters),
            readout=readout,
            square_half_hidden_state=metadata["square_half_hidden_state"],
        )
