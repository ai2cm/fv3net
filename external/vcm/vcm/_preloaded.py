from typing import Sequence, Callable, Any
import collections.abc
import random
import concurrent.futures


class Preloaded(collections.abc.Iterator):
    """
    Iterator for data which asynchronously pre-loads the next set of output.
    """

    def __init__(
        self,
        filenames: Sequence[str],
        loader_function: Callable[[str], Any],
        shuffle: bool = True,
    ):
        """
        Args:
            filenames: sequence to be passed to loader function
            shuffle: if True, return items in random order
        """
        self.loader_function = loader_function
        self._filenames = list(filenames)
        self._shuffle = shuffle
        self._maybe_shuffle()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._idx = 0
        self._load_thread = None

    def _start_load(self):
        if self._idx < len(self._filenames):
            self._load_thread = self._executor.submit(
                self.loader_function, self._filenames[self._idx],
            )

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration()
        else:
            arrays = self._load_thread.result()
            self._load_thread = None
            self._idx += 1
            if self._idx < len(self):
                self._start_load()
            return arrays

    def _maybe_shuffle(self):
        if self._shuffle:
            # new shuffled order each time we iterate
            random.shuffle(self._filenames)

    def __iter__(self) -> "Preloaded":
        self._idx = 0
        self._maybe_shuffle()
        if self._load_thread is None:
            self._start_load()
        return self

    def __len__(self):
        return len(self._filenames)
