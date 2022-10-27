import abc
import dacite
import dataclasses
import fsspec
import io
import joblib
import numpy as np
import yaml

from .readout import ReservoirComputingReadout
from .reservoir import Reservoir
from .config import ReservoirHyperparameters


class ImperfectModel(abc.ABC):
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def predict(self, input: np.ndarray) -> np.ndarray:
        """
        Predict one reservoir computing step ahead.
        If the imperfect model takes shorter timesteps than the reservoir model,
        this should return the imperfect model's prediction at the next reservoir step.
        """
        pass


class ReservoirComputingModel:
    _READOUT_NAME = "readout.pkl"
    _METADATA_NAME = "metadata.bin"

    def __init__(
        self, reservoir: Reservoir, readout: ReservoirComputingReadout,
    ):
        self.reservoir = reservoir
        self.readout = readout

    def predict(self, **kwargs):
        prediction = self.readout.predict(self.reservoir.state).reshape(-1)
        self.reservoir.increment_state(prediction)
        return prediction

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)
        mapper = fs.get_mapper(path)

        mapper[self._READOUT_NAME] = self.readout.dumps()
        metadata = {
            "reservoir_hyperparameters": dataclasses.asdict(
                self.reservoir.hyperparameters
            )
        }
        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingModel":
        """Load a model from a remote path"""
        mapper = fsspec.get_mapper(path)

        f = io.BytesIO(mapper[cls._READOUT_NAME])
        readout_components = joblib.load(f)
        readout = ReservoirComputingReadout(**readout_components)
        metadata = yaml.safe_load(mapper[cls._METADATA_NAME])

        reservoir_hyperparameters = dacite.from_dict(
            ReservoirHyperparameters, metadata["reservoir_hyperparameters"]
        )

        return cls(reservoir=Reservoir(reservoir_hyperparameters), readout=readout,)


class HybridReservoirComputingModel(ReservoirComputingModel):
    _IMPERFECT_MODEL_NAME = "hybrid_rc_imperfect_model.bin"

    def __init__(
        self,
        reservoir: Reservoir,
        readout: ReservoirComputingReadout,
        imperfect_model: ImperfectModel,
    ):
        self.reservoir = reservoir
        self.readout = readout
        self.imperfect_model = imperfect_model

    def predict(self, input_state):
        imperfect_prediction = self.imperfect_model.predict(input_state)
        readout_input = np.hstack([self.reservoir.state, imperfect_prediction])
        rc_prediction = self.readout.predict(readout_input).reshape(-1)
        self.reservoir.increment_state(rc_prediction)
        return rc_prediction

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        super().dump(path)

        # TODO: If we move on to non- toy model cases, the dumping of the
        # imperfect model should be handled by the child classes.
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)
        mapper = fs.get_mapper(path)
        f = io.BytesIO()
        joblib.dump(self.imperfect_model, f)
        mapper[self._IMPERFECT_MODEL_NAME] = f.getvalue()

    @classmethod
    def load(cls, path):
        mapper = fsspec.get_mapper(path)

        f_readout = io.BytesIO(mapper[cls._READOUT_NAME])
        readout_components = joblib.load(f_readout)
        readout = ReservoirComputingReadout(**readout_components)
        metadata = yaml.safe_load(mapper[cls._METADATA_NAME])

        reservoir_hyperparameters = dacite.from_dict(
            ReservoirHyperparameters, metadata["reservoir_hyperparameters"]
        )
        f_imperfect_model = io.BytesIO(mapper[cls._IMPERFECT_MODEL_NAME])
        imperfect_model = joblib.load(f_imperfect_model)

        return cls(
            reservoir=Reservoir(reservoir_hyperparameters),
            readout=readout,
            imperfect_model=imperfect_model,
        )
