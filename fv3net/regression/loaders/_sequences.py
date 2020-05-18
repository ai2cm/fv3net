import collections
from typing import Callable, Sequence, Iterable


class FunctionOutputSequence(collections.abc.Sequence):
    """A wrapper over a sequence of function arguments passed into a function."""

    def __init__(self, func: Callable, args_sequence: Sequence[Iterable]):
        """
        Args:
            func: the function to call
            args_sequence: a sequence of argument iterables
        Returns:
            result_sequence: a sequence of function results
        """
        self._func = func
        self._args = args_sequence

    def __getitem__(self, item):
        return self._func(*self._args[item])

    def __len__(self):
        return len(self._args)
