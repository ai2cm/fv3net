import contextlib
import pytest


@contextlib.contextmanager
def no_warning(*arg):
    with pytest.warns(*arg) as record:
        yield

    assert len(record) == 0
