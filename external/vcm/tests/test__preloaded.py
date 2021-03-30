from typing import Sequence, Tuple, Dict, Any
import random
import vcm
import pytest
import time


class CallRecord:
    def __init__(self, args, kwargs):
        self.args: Tuple[Any] = args
        self.kwargs: Dict[str, Any] = kwargs


class DummyFunction:
    def __init__(self):
        self.call_history: Sequence[CallRecord] = []

    def __call__(self, *args, **kwargs) -> CallRecord:
        self.call_history.append(CallRecord(args, kwargs))
        return CallRecord(args, kwargs)


@pytest.fixture(params=[0, 1, 5])
def n_filenames(request):
    return request.param


@pytest.fixture
def filenames(n_filenames):
    return [f"{i}.ext" for i in range(n_filenames)]


@pytest.fixture
def loader_function():
    return DummyFunction()


@pytest.fixture
def preloaded(filenames, loader_function):
    return vcm.Preloaded(filenames, loader_function)


def test_loads_all_files(filenames, preloaded, loader_function):
    for _ in preloaded:
        pass
    loaded_files = [record.args[0] for record in loader_function.call_history]
    assert set(loaded_files) == set(filenames)


def test_loads_all_files_once(filenames, preloaded, loader_function):
    for _ in preloaded:
        pass
    assert len(loader_function.call_history) == len(filenames)
    loaded_files = [record.args[0] for record in loader_function.call_history]
    assert set(loaded_files) == set(filenames)


@pytest.mark.parametrize("n_filenames", [5])
def test_loads_next_file(filenames, preloaded, loader_function):
    for i, record in enumerate(preloaded):
        if i < len(filenames) - 1:  # no next file for the last file
            time.sleep(0.01)
            n_iterations = i + 1
            assert len(loader_function.call_history) == n_iterations + 1
            # we only pass args, so no need to compare kwargs
            assert loader_function.call_history[-1].args != record.args
            assert loader_function.call_history[-2].args == record.args


@pytest.mark.parametrize("n_filenames", [10])
def test_loads_in_different_order_from_given(filenames, preloaded, loader_function):
    random.seed(0)
    for _ in preloaded:
        pass
    loaded_files = [record.args[0] for record in loader_function.call_history]
    assert tuple(loaded_files) != tuple(filenames)


@pytest.mark.parametrize("n_filenames", [10])
def test_loads_in_different_order_each_loop(preloaded, loader_function):
    random.seed(0)
    for _ in preloaded:
        pass
    first_loaded_files = [record.args[0] for record in loader_function.call_history]
    for _ in preloaded:
        pass
    second_loaded_files = [record.args[0] for record in loader_function.call_history]
    assert tuple(first_loaded_files) != tuple(second_loaded_files)
