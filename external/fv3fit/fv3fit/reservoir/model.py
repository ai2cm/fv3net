from .readout import ReservoirComputingReadout
from .reservoir import Reservoir


class ReservoirComputingModel:
    _RESERVOIR_SUBDIR = "reservoir"

    def __init__(
        self, reservoir: Reservoir, readout: ReservoirComputingReadout,
    ):
        self.reservoir = reservoir
        self.readout = readout

    def predict(self):
        prediction = self.readout.predict(self.reservoir.state).reshape(-1)
        self.reservoir.increment_state(prediction)
        return prediction

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        self.readout.dump(path)
        self.reservoir.dump(f"{path}/{self._RESERVOIR_SUBDIR}")

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingModel":
        """Load a model from a remote path"""
        readout = ReservoirComputingReadout.load(path)
        reservoir = Reservoir.load(f"{path}/{cls._RESERVOIR_SUBDIR}")
        return cls(reservoir=reservoir, readout=readout,)
