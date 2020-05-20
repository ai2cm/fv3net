import warnings
import pytest

from vcm import testing


def test_no_warning():
    with pytest.raises(AssertionError):
        with testing.no_warning(None):
            warnings.warn("Warning")
