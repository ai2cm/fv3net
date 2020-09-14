import collections
from copy import deepcopy
from functools import partial
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


def shuffle(
    sequence: Sequence[Any], seed: Optional[int] = None
) -> FunctionOutputSequence:
    """
    Shuffle a sequence by creating a new FunctionOutputSequence
    with shuffled indices as arguments.  Preserves potentially lazy
    operations on input sequence by only shuffling potential __getitem__
    arguments.

    Args:
        sequence:  Input sequence to have access indices shuffled
        seed: Seed for random number generator used for shuffling
    Returns:
        A new shuffled sequence
    """
    random = RandomState(seed)
    seq_len = len(sequence)
    shuffled = random.choice(seq_len, size=seq_len, replace=False).tolist()
    func = partial(_simple_getitem, sequence)
    return FunctionOutputSequence(func, shuffled)


def _simple_getitem(sequence: Sequence[Any], item: Union[int, slice]):
    return sequence[item]
