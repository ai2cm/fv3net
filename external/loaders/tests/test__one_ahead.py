import loaders
import pytest
import time
from unittest.mock import Mock, call


@pytest.fixture(params=[0, 1, 5])
def n_filenames(request):
    return request.param


@pytest.fixture
def filenames(n_filenames):
    return [f"{i}.ext" for i in range(n_filenames)]


@pytest.fixture
def loader_function():
    return Mock()


def test_loads_all_files(filenames, loader_function):
    one_ahead = loaders.OneAheadIterator(filenames, loader_function)
    for _ in one_ahead:
        pass
    expected_calls = [call(fname) for fname in filenames]
    loader_function.assert_has_calls(expected_calls, any_order=True)


@pytest.mark.parametrize("n_filenames", [5])
def test_loads_next_file(filenames, loader_function):
    one_ahead = loaders.OneAheadIterator(filenames, loader_function)
    expected_calls = [call(fname) for fname in filenames]
    for i, _ in enumerate(one_ahead):
        if i < len(filenames) - 1:  # no next file for the last file
            time.sleep(0.01)
            n_expected_calls = i + 1
            loader_function.assert_has_calls(expected_calls[:n_expected_calls])
