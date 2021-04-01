from typing import Sequence, Callable, Any
import collections.abc
import concurrent.futures


class OneAheadIterator(collections.abc.Iterator):
    """
    Iterator which asynchronously pre-computes the next output in a thread.
    """

    def __init__(
        self, args: Sequence[str], function: Callable[[Any], Any],
    ):
        """
        Args:
            args: sequence to be passed to loader function
            function: single-argument function to receive arguments
        """
        self.function = function
        self._args = args
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._idx = 0
        self._load_thread = None

    def _start_load(self):
        if self._idx < len(self._args):
            self._load_thread = self._executor.submit(
                self.function, self._args[self._idx],
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

    def __iter__(self) -> "OneAheadIterator":
        self._idx = 0
        if self._load_thread is None:
            self._start_load()
        return self

    def __len__(self):
        return len(self._args)

    def __del__(self):
        # check necessary in case exceptions occur before this is defined
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
        super().__del__()
