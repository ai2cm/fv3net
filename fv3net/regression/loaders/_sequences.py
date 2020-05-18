import collections
from typing import Callable, Sequence, Iterable


class FunctionOutputSequence(collections.abc.Sequence):
    """A wrapper over a sequence of function arguments passed into a function."""

    def __init__(self, func: Callable, args_sequence: Sequence[Iterable]):
        """
        Args:
            func: the function to call, which takes in one argument
            args_sequence: a sequence of arguments
        Returns:
            result_sequence: a sequence of function results
        """
        if not isinstance(args_sequence, collections.Sequence):
            raise TypeError(f'args_sequence must be a sequence, got {args_sequence}')
        self._func = func
        self._args = args_sequence

    def __getitem__(self, item):
        return self._func(self._args[item])

    def __len__(self):
        return len(self._args)
