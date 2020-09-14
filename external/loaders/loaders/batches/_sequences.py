import collections
from copy import deepcopy
from numpy.random import RandomState
from typing import (
    Callable,
    Sequence,
    MutableMapping,
    TypeVar,
    Hashable,
    Any,
    Optional,
    Union,
)

T = TypeVar("T")


class FunctionOutputSequence(Sequence[T]):
    """A wrapper over a sequence of function arguments passed into a function.

    Attributes:
        attrs: a dictionary of metadata.
    """

    attrs: MutableMapping[Hashable, Any]

    def __init__(self, func: Callable[..., T], args_sequence: Sequence[Any]):
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

    def __getitem__(self, item: Union[int, slice]) -> T:

        if isinstance(item, int):
            return self._func(self._args[item])
        elif isinstance(item, slice):
            return self._slice_selection(item)
        else:
            TypeError(f"Invalid argument type of {type(item)} passed into __getitem__.")

    def _slice_selection(self, selection: slice):
        seq = self.__class__(self._func, self._args[selection])
        seq.attrs.update(deepcopy(self.attrs))
        return seq

    def __len__(self) -> int:
        return len(self._args)


class Shuffle(FunctionOutputSequence):
    def __init__(self, sequence: Sequence[Any], seed: Optional[int] = None):

        self._random = RandomState(seed)
        self._sequence = sequence

        seq_len = len(sequence)
        shuffled = self._random.choice(seq_len, size=seq_len, replace=False)

        super().__init__(self._simple_load, shuffled)

    def _simple_load(self, item: int):
        return self._sequence[item]
