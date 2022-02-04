import os
import glob
import joblib
import collections.abc
from copy import deepcopy
from functools import partial
import numpy as np
from typing import (
    Callable,
    Sequence,
    MutableMapping,
    TypeVar,
    Hashable,
    Any,
    Union,
)

T = TypeVar("T")


class BaseSequence(Sequence[T]):
    def local(self, path: str, n_jobs: int = 4) -> "Local":
        """Download a sequence of xarray objects to a local path

        Args:
            path: local directory, will be created if not existing
            n_jobs: parallelism
        """
        return to_local(self, path=path, n_jobs=n_jobs)

    def _save_item(self, path: str, i: int):
        item = self[i]
        path = os.path.join(path, "%05d.pkl" % i)
        Local.dump(item, path)

    def take(self, n: int) -> "Take":
        """Return a sequence consisting of the first n elements
        """
        return Take(self, n)

    def map(self, func) -> "Map":
        """Map a function over the elements of this sequence
        """
        return Map(func, self)


class Take(BaseSequence[T]):
    def __init__(self, parent_seq, n):
        self._seq = parent_seq
        self.n = n

    def __getitem__(self, i):
        if i < len(self):
            return self._seq[i]
        else:
            raise IndexError()

    def __len__(self):
        return self.n


class Local(BaseSequence[T]):
    def __init__(self, path: str):
        self.path = path

    @property
    def files(self):
        return sorted(glob.glob(os.path.join(self.path, "*.pkl")))

    @classmethod
    def dump(cls, dataset, path):
        try:
            joblib.dump(dataset.load(), path)
        except AttributeError:
            joblib.dump(dataset, path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        slice_value = self.files[i]
        if isinstance(slice_value, str):
            return joblib.load(slice_value)
        else:
            return [joblib.load(file) for file in slice_value]


def to_local(sequence: Sequence[T], path: str, n_jobs: int = 4) -> Local[T]:
    """
    Download a sequence of pickleable objects to a local path.

    Args:
        sequence: pickleable objects to dump locally
        path: local directory, will be created if not existing
        n_jobs: how many threads to use when dumping objects to file
    
    Returns:
        local_sequence
    """
    os.makedirs(path, exist_ok=True)

    def save_item(path: str, i: int):
        item = sequence[i]
        path = os.path.join(path, "%05d.pkl" % i)
        Local.dump(item, path)

    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(save_item)(path, i) for i in range(len(sequence))
    )
    return Local(os.path.abspath(path))


class Map(BaseSequence[T]):
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
        if not isinstance(args_sequence, collections.abc.Sequence):
            raise TypeError(f"args_sequence must be a sequence, got {args_sequence}")
        self._func = func
        self._args = args_sequence
        self.attrs = {}

    def __getitem__(self, item: Union[int, slice]):

        if isinstance(item, int):
            return self._func(self._args[item])
        elif isinstance(item, slice):
            return self._slice_selection(item)
        else:
            TypeError(f"Invalid argument type of {type(item)} passed into __getitem__.")

    def _slice_selection(self, selection: slice):
        seq = Map(self._func, self._args[selection])
        seq.attrs.update(deepcopy(self.attrs))
        return seq

    def __len__(self) -> int:
        return len(self._args)


def shuffle(sequence: Sequence[T]) -> Map[T]:
    """Lazily shuffle a sequence. Uses numpy.random for randomness.

    Args:
        sequence:  Input sequence to have access indices shuffled
    Returns:
        A new shuffled sequence
    """
    seq_len = len(sequence)
    shuffled = np.random.choice(seq_len, size=seq_len, replace=False).tolist()
    func = partial(_simple_getitem, sequence)
    return Map(func, shuffled)


def _simple_getitem(sequence: Sequence[Any], item: Union[int, slice]):
    return sequence[item]
