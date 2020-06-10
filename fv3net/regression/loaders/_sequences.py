import collections
from typing import Callable, Sequence, Iterable, Mapping, TypeVar

T = TypeVar("T")


class FunctionOutputSequence(Sequence[T]):
    """A wrapper over a sequence of function arguments passed into a function.

    Attributes:
        attrs: a dictionary of metadata.
    """

    attrs: Mapping

    def __init__(self, func: Callable[..., T], args_sequence: Sequence[Iterable]):
        """
        Args:
            func: the function to call, which takes in one argument
            args_sequence: a sequence of arguments
        Returns:
            result_sequence: a sequence of function results
        """
        if not isinstance(args_sequence, collections.Sequence):
            raise TypeError(f"args_sequence must be a sequence, got {args_sequence}")
        self._func = func
        self._args = args_sequence
        self.attrs = {}

    def __getitem__(self, item: int) -> T:
        return self._func(self._args[item])

    def __len__(self) -> int:
        return len(self._args)
