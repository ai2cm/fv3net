import abc
from typing import Dict
from loaders.typing import (
    Mapper,
    MapperFunction,
    Batches,
    BatchesFunction,
    BatchesFromMapperFunction,
)
import dataclasses
import dacite


MAPPER_FUNCTIONS: Dict[str, MapperFunction] = {}


def register_mapper_function(func: MapperFunction):
    MAPPER_FUNCTIONS[func.__name__] = func
    return func


BATCHES_FUNCTIONS: Dict[str, BatchesFunction] = {}


def register_batches_function(func):
    BATCHES_FUNCTIONS[func.__name__] = func
    return func


BATCHES_FROM_MAPPER_FUNCTIONS: Dict[str, BatchesFromMapperFunction] = {}


def register_batches_from_mapper_function(func):
    BATCHES_FROM_MAPPER_FUNCTIONS[func.__name__] = func
    return func


@dataclasses.dataclass
class MapperConfig:

    data_path: str
    mapper_function: str
    mapper_kwargs: dict

    def load_mapper(self) -> Mapper:
        """
        Args:
            config: data configuration

        Returns:
            Sequence of datasets according to configuration
        """
        mapping_func = MAPPER_FUNCTIONS[self.mapper_function]
        return mapping_func(self.data_path, **self.mapper_kwargs)


class BatchesLoader(abc.ABC):
    @abc.abstractmethod
    def load_batches(self, variables) -> Batches:
        """
        Args:
            config: data configuration

        Returns:
            Sequence of datasets according to configuration
        """
        pass

    @classmethod
    def from_dict(cls, kwargs) -> "BatchesLoader":
        try:
            return dacite.from_dict(data_class=cls, data=kwargs)
        except (TypeError, AttributeError):
            pass
        for subclass in cls.__subclasses__():
            try:
                return dacite.from_dict(data_class=subclass, data=kwargs)
            except (TypeError, AttributeError, dacite.exceptions.MissingValueError):
                pass
        raise ValueError("invalid BatchesLoader dictionary")


@dataclasses.dataclass
class BatchesFromMapperConfig(BatchesLoader):

    mapper_config: MapperConfig
    batches_function: str
    batches_kwargs: dict

    def load_mapper(self) -> Mapper:
        return self.mapper_config.load_mapper()

    def load_batches(self, variables) -> Batches:
        """
        Args:
            variables: names of variables to include in dataset

        Returns:
            Sequence of datasets according to configuration
        """
        mapper = self.mapper_config.load_mapper()
        batches_function = BATCHES_FROM_MAPPER_FUNCTIONS[self.batches_function]
        return batches_function(mapper, list(variables), **self.batches_kwargs,)


@dataclasses.dataclass
class BatchesConfig(BatchesLoader):
    """Convenience wrapper for model training data.

    Attrs:
        data_path: location of training data to be loaded by batch function
        batches_function: name of function from `fv3fit.batches` to use for
            loading batched data
        batches_kwargs: keyword arguments to pass to batch function
    """

    data_path: str
    batches_function: str
    batches_kwargs: dict

    def load_batches(self, variables) -> Batches:
        """
        Args:
            variables: names of variables to include in dataset

        Returns:
            Sequence of datasets according to configuration
        """
        batches_function = BATCHES_FUNCTIONS[self.batches_function]
        return batches_function(self.data_path, list(variables), **self.batches_kwargs,)
