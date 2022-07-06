import abc
import tensorflow as tf
from typing import Optional, Sequence, Type, List
import dacite

_TFDATASET_LOADERS: List[Type["TFDatasetLoader"]] = []


class TFDatasetLoader(abc.ABC):
    @abc.abstractmethod
    def get_data(
        self, local_download_path: Optional[str], variable_names: Sequence[str],
    ) -> tf.data.Dataset:
        """
        Args:
            local_download_path: if provided, cache data locally at this path
            variable_names: names of variables to include when loading data
        Returns:
            dataset containing requested variables, each record is a mapping from
                variable name to variable value, and each value is a tensor whose
                first dimension is the batch dimension
        """
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "TFDatasetLoader":
        raise NotImplementedError("must be implemented by subclass")


def register_tfdataset_loader(loader_class: Type[TFDatasetLoader]):
    """
    Register a TFDatasetLoader subclass as a factory for TFDatasetLoaders.
    """
    global _TFDATASET_LOADERS
    _TFDATASET_LOADERS.append(loader_class)
    return loader_class


def tfdataset_loader_from_dict(d: dict) -> TFDatasetLoader:
    for cls in _TFDATASET_LOADERS:
        try:
            return cls.from_dict(d)
        except (
            TypeError,
            ValueError,
            AttributeError,
            dacite.exceptions.MissingValueError,
            dacite.exceptions.UnexpectedDataError,
        ):
            pass
    raise ValueError("invalid TFDatasetLoader dictionary")
