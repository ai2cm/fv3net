import contextlib
import pytest


@contextlib.contextmanager
def no_warning(*args):
    """Raise error if any errors occur. Takes the same arguments as 
    ``pytest.warns``.

    Example:

        >>> import warnings
        >>> from vcm.testing import no_warning
        >>> with no_warning(UserWarning):
        ...     warnings.warn(UserWarning("A warning"))
        ...
        Traceback (most recent call last):
        File "<ipython-input-9-c178a20fa539>", line 2, in <module>
            warnings.warn(UserWarning("A warning"))
        File "/Users/noah/.pyenv/versions/miniconda3-latest/envs/fv3net/lib/python3.7/contextlib.py", line 119, in __exit__
            next(self.gen)
        File "/Users/noah/workspace/fv3net/external/vcm/vcm/testing.py", line 14, in no_warning
            assert len(record) == 0
        AssertionError

    """  # noqa
    with pytest.warns(*args) as record:
        yield

    assert len(record) == 0
