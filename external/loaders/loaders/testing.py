from loaders import _config
import loaders.batches._batch
import contextlib
import xarray as xr
import unittest.mock


@contextlib.contextmanager
def registration_context(registration_dict):
    """
    A context manager that provides a clean slate for registering functions,
    and restores the registration state when exiting.
    """
    original_functions = {}
    original_functions.update(registration_dict)
    registration_dict.clear()
    try:
        yield
    finally:
        registration_dict.clear()
        registration_dict.update(original_functions)


@contextlib.contextmanager
def mapper_context():
    """
    A context manager that provides a clean slate for registering mapper functions,
    and restores the registration state when exiting.
    """
    with registration_context(_config.mapper_functions):
        mock_mapper = {"key": xr.Dataset()}
        mock_mapper_function = unittest.mock.MagicMock(return_value=mock_mapper)
        mock_mapper_function.__name__ = "mock_mapper_function"
        _config.mapper_functions.register(mock_mapper_function)
        yield mock_mapper_function


@contextlib.contextmanager
def batches_context():
    """
    A context manager that provides a clean slate for registering batches functions,
    and restores the registration state when exiting.
    """
    with registration_context(_config.batches_functions):
        mock_batches = [xr.Dataset()]
        mock_batches_function = unittest.mock.MagicMock(return_value=mock_batches)
        mock_batches_function.__name__ = "mock_batches_function"
        _config.batches_functions.register(mock_batches_function)
        yield mock_batches_function


@contextlib.contextmanager
def batches_from_mapper_context():
    """
    A context manager that provides a clean slate for registering
    batches from mapper functions, and restores the registration state when exiting.
    """
    mock = unittest.mock.MagicMock(spec=loaders.batches._batch.batches_from_mapper)
    with unittest.mock.patch("loaders.batches._batch.batches_from_mapper", new=mock):
        yield mock
