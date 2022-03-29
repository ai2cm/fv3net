import abc
from typing import Dict, Optional, Sequence, TypeVar, Callable
from loaders.typing import (
    Mapper,
    Batches,
)
import dataclasses
import dacite


RT = TypeVar("RT")


class FunctionRegister(Dict[str, Callable[..., RT]]):
    def register(self, func: Callable[..., RT]) -> Callable[..., RT]:
        self[func.__name__] = func
        return func

    def __repr__(self):
        return str(sorted(list(self.keys())))


mapper_functions: FunctionRegister[Mapper] = FunctionRegister()
batches_functions: FunctionRegister[Batches] = FunctionRegister()


@dataclasses.dataclass
class MapperConfig:
    """Configuration for the use of mapper loading functions.

    Attributes:
        function: name of function to use for loading batched data,
            can take any value in the keys of `loaders.mapper_functions`
        kwargs: keyword arguments to pass to mapper function
    """

    function: str
    kwargs: dict

    def load_mapper(self) -> Mapper:
        """
        Returns:
            Sequence of mappers according to configuration
        """
        mapping_func = mapper_functions[self.function]
        return mapping_func(**self.kwargs)

    def __post_init__(self):
        if self.function not in mapper_functions:
            raise ValueError(
                f"Invalid mapper function {self.function}, "
                f"must be one of {list(mapper_functions.keys())}"
            )


class BatchesLoader(abc.ABC):
    """
    Abstract base class for configuration classes that load batches. See
    subclasses for concrete implementations you can use in configuration files.
    """

    @abc.abstractmethod
    def load_batches(self, variables: Optional[Sequence[str]] = None) -> Batches:
        """
        Args:
            variables: if given, these variables are guaranteed to be present in
                the returned batches, or an exception will be raised

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
class BatchesConfig(BatchesLoader):
    """Configuration for the use of batch loading functions.

    Attributes:
        function: name of function to use for loading batched data,
            can take any value in the keys of `loaders.batches_functions`
        kwargs: keyword arguments to pass to batches function
    """

    function: str
    kwargs: dict

    def load_batches(self, variables: Optional[Sequence[str]] = None) -> Batches:
        """
        Args:
            variables: if given, these variables are guaranteed to be present in
                the returned batches, or an exception will be raised

        Returns:
            Sequence of datasets according to configuration
        """
        if variables is None:
            variables = []
        batches_function = batches_functions[self.function]
        kwargs = {**self.kwargs}
        kwargs["variable_names"] = list(kwargs.get("variable_names", []))
        kwargs["variable_names"].extend(variables)
        return batches_function(**kwargs)

    def __post_init__(self):
        if self.function not in batches_functions:
            raise ValueError(
                f"Invalid batches function {self.function}, "
                f"must be one of {list(batches_functions.keys())}"
            )
