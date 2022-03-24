import abc
import numpy as np
from typing import BinaryIO, Type, Sequence
import io
import yaml


class NormalizeTransform(abc.ABC):
    @abc.abstractproperty
    def kind(self) -> str:
        pass

    @abc.abstractmethod
    def normalize(self, y: np.ndarray):
        pass

    @abc.abstractmethod
    def denormalize(self, y: np.ndarray):
        pass

    @abc.abstractmethod
    def dump(self, f: BinaryIO):
        pass

    def dumps(self) -> bytes:
        f = io.BytesIO()
        self.dump(f)
        return f.getvalue()

    @classmethod
    @abc.abstractmethod
    def load(cls, f: BinaryIO):
        pass


class StandardScaler(NormalizeTransform):

    kind: str = "standard"

    def __init__(self, std_epsilon: np.float64 = 1e-12):
        """Standard scaler normalizer: normalizes via (x-mean)/std

        Args:
            std_epsilon: A small value that is added to the standard deviation
                of each variable to be scaled, such that no variables (even those
                that are constant across samples) are unable to be scaled due to
                having zero standard deviation. Defaults to 1e-12.
        """
        self.mean = None
        self.std = None
        self.std_epsilon: np.float64 = std_epsilon

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0).astype(np.float64)
        self.std = np.std(data, axis=0).astype(np.float64) + self.std_epsilon

    def normalize(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler.fit must be called before normalize.")
        return (data - self.mean) / self.std

    def denormalize(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler.fit must be called before denormalize.")
        return data * self.std + self.mean

    def dump(self, f: BinaryIO):
        data = {}  # type: ignore
        if self.mean is not None:
            data["mean"] = self.mean
        if self.std is not None:
            data["std"] = self.std
        return np.savez(f, **data)

    @classmethod
    def load(cls, f: BinaryIO):
        data = np.load(f)
        scaler = cls()
        scaler.mean = data.get("mean")
        scaler.std = data.get("std")
        return scaler


class ManualScaler(NormalizeTransform):

    kind: str = "manual"

    def __init__(self, scales):
        self.scales = scales

    def normalize(self, y: np.ndarray):
        return y * self.scales

    def denormalize(self, y: np.ndarray):
        return y / self.scales

    def dump(self, f: BinaryIO):
        data = {}
        if self.scales is not None:
            data["scales"] = self.scales
        return np.savez(f, **data)

    @classmethod
    def load(cls, f: BinaryIO):
        data = np.load(f)
        scales = data.get("scales")
        scaler = cls(scales)
        return scaler


scalers: Sequence[Type[NormalizeTransform]] = [StandardScaler, ManualScaler]


def dumps(scaler: NormalizeTransform) -> str:
    """Dump scaler object to string
    """
    return yaml.safe_dump((scaler.kind, scaler.dumps()))


def loads(b: str) -> NormalizeTransform:
    """Load scaler from string
    """
    class_name, data = yaml.safe_load(b)
    f = io.BytesIO(data)
    for scaler_cls in scalers:
        if class_name == scaler_cls.kind:
            return scaler_cls.load(f)

    raise NotImplementedError(f"Cannot load {class_name} scaler")
