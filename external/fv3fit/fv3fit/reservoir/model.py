import fsspec
from os.path import basename, normpath
from typing import Optional, Iterable, Hashable
import yaml

from fv3fit import Predictor
from fv3fit._shared.scaler import StandardScaler
from fv3fit._shared import io
from .readout import ReservoirComputingReadout
from .reservoir import Reservoir


def _exists_in_dir(file_name, dir):
    fs = fsspec.get_fs_token_paths(dir)[0]
    contents = [basename(normpath(p)) for p in fs.ls(dir)]
    return file_name in contents


@io.register("pure-reservoir")
class ReservoirComputingModel(Predictor):
    _SCALER_NAME = "scaler.npz"
    _RESERVOIR_SUBDIR = "reservoir"
    _METADATA_NAME = "reservoir_computing_model_metadata.bin"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        reservoir: Reservoir,
        readout: ReservoirComputingReadout,
        scaler: Optional[StandardScaler] = None,
    ):
        """ Wraps a trained readout and its corresponding reservoir to make
        predictions. If training was done with normalized data, the scaler
        should be provided so that it can denormalize the output prediction.

        Args:
            input_variables: inputs
            output_variables: outputs, should be a subset of input_variables
            reservoir : reservoir
            readout : trained readout
            scaler : StandardScaler that was used to normalize the data during
                training.
        """
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.reservoir = reservoir
        self.readout = readout
        self.scaler = scaler

    def predict(self):
        prediction = self.readout.predict(self.reservoir.state).reshape(-1)
        if self.scaler is not None:
            return self.scaler.denormalize(prediction)
        else:
            return prediction

    def increment_state(self, prediction_with_overlap):
        self.reservoir.increment_state(prediction_with_overlap)

    def synchronize_reservoir(self, synchronization_time_series):
        if self.scaler is not None:
            synchronization_time_series = self.scaler.normalize(
                synchronization_time_series
            )
        self.reservoir.synchronize(synchronization_time_series)

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        self.readout.dump(path)
        self.reservoir.dump(f"{path}/{self._RESERVOIR_SUBDIR}")
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        metadata = {
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
        }
        mapper = fs.get_mapper(path)
        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")
        if self.scaler is not None:
            with fs.open(f"{path}/{self._SCALER_NAME}", "wb") as f:
                self.scaler.dump(f)

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingModel":
        """Load a model from a remote path"""
        readout = ReservoirComputingReadout.load(path)
        reservoir = Reservoir.load(f"{path}/{cls._RESERVOIR_SUBDIR}")

        scaler = None
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        if _exists_in_dir(cls._SCALER_NAME, path):
            with fs.open(f"{path}/{cls._SCALER_NAME}", "rb") as f:
                scaler = StandardScaler.load(f)
        mapper = fsspec.get_mapper(path)
        metadata = yaml.safe_load(mapper[cls._METADATA_NAME])
        return cls(
            input_variables=metadata["input_variables"],
            output_variables=metadata["output_variables"],
            reservoir=reservoir,
            readout=readout,
            scaler=scaler,
        )
