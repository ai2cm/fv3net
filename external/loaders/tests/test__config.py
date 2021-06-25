import pytest
import unittest.mock
import loaders
import contextlib
import xarray as xr
import tempfile
import os
import yaml
import dataclasses


@contextlib.contextmanager
def registration_context(registration_dict):
    """
    A context manager that provides a clean slate for registering functions,
    and restores the registration state when exiting.
    """
    original_functions = {}
    original_functions.update(registration_dict)
    registration_dict.clear()
    yield
    registration_dict.clear()
    registration_dict.update(original_functions)


@contextlib.contextmanager
def mapper_context():
    with registration_context(loaders._config.mapper_functions):
        yield


@contextlib.contextmanager
def batches_context():
    with registration_context(loaders._config.batches_functions):
        yield


@contextlib.contextmanager
def batches_from_mapper_context():
    with registration_context(loaders._config.batches_from_mapper_functions):
        yield


def test_load_mapper():
    with mapper_context():
        mock_mapper = {"key": xr.Dataset()}
        mock_mapper_function = unittest.mock.MagicMock(return_value=mock_mapper)
        mock_mapper_function.__name__ = "mock_mapper_function"
        loaders._config.mapper_functions.register(mock_mapper_function)
        assert "mock_mapper_function" in loaders._config.mapper_functions
        data_path = "test/data/path"
        mapper_kwargs = {"arg1": "value1", "arg2": 2}
        config = loaders._config.MapperConfig(
            data_path=data_path,
            mapper_function="mock_mapper_function",
            mapper_kwargs=mapper_kwargs,
        )
        result = config.load_mapper()
        assert result is mock_mapper
        mock_mapper_function.assert_called_once_with(data_path, **mapper_kwargs)


def test_expected_mapper_functions_exist():
    # this test exists to ensure we don't accidentally remove functions we
    # currently use in configs, if you are deleting an option we no longer use
    # you can delete it here
    for expected in (
        "open_nudge_to_obs",
        "open_nudge_to_fine",
        "open_fine_res_apparent_sources",
        "open_high_res_diags",
        "open_fine_resolution_nudging_hybrid",
        "open_nudge_to_fine_multiple_datasets",
    ):
        assert expected in loaders._config.mapper_functions


def test_expected_batches_functions_exist():
    # this test exists to ensure we don't accidentally remove functions we
    # currently use in configs, if you are deleting an option we no longer use
    # you can delete it here
    for expected in (
        "batches_from_geodata",
        "batches_from_serialized",
        "diagnostic_batches_from_geodata",
    ):
        assert expected in loaders._config.batches_functions


def test_expected_batches_from_mapper_functions_exist():
    # this test exists to ensure we don't accidentally remove functions we
    # currently use in configs, if you are deleting an option we no longer use
    # you can delete it here
    for expected in ("batches_from_mapper",):
        assert expected in loaders._config.batches_from_mapper_functions


def assert_registered_functions_are_public(function_register):
    missing_functions = []
    for name, func in function_register.items():
        if getattr(loaders, name, None) is not func:
            missing_functions.append(name)
    assert (
        len(missing_functions) == 0
    ), "registered functions are public and should exist in root namespace"


def test_registered_mapper_functions_are_public():
    assert_registered_functions_are_public(loaders.mapper_functions)


def test_registered_batches_functions_are_public():
    assert_registered_functions_are_public(loaders.batches_functions)


def test_registered_batches_from_mapper_functions_are_public():
    assert_registered_functions_are_public(loaders.batches_from_mapper_functions)


def test_load_batches():
    with batches_context():
        mock_batches = [xr.Dataset()]
        mock_batches_function = unittest.mock.MagicMock(return_value=mock_batches)
        mock_batches_function.__name__ = "mock_batches_function"
        loaders._config.batches_functions.register(mock_batches_function)
        assert "mock_batches_function" in loaders._config.batches_functions
        data_path = "test/data/path"
        variables = ["var1"]
        batches_kwargs = {"arg1": "value1", "arg2": 2}
        config = loaders._config.BatchesConfig(
            data_path=data_path,
            batches_function="mock_batches_function",
            batches_kwargs=batches_kwargs,
        )
        result = config.load_batches(variables=variables)
        assert result is mock_batches
        mock_batches_function.assert_called_once_with(
            data_path, variables, **batches_kwargs
        )


def test_load_batches_from_mapper():
    with batches_from_mapper_context(), mapper_context():
        mock_batches = [xr.Dataset()]
        mock_batches_function = unittest.mock.MagicMock(return_value=mock_batches)
        mock_batches_function.__name__ = "mock_batches_function"
        loaders._config.batches_from_mapper_functions.register(mock_batches_function)
        mock_mapper = {"key": xr.Dataset()}
        mock_mapper_function = unittest.mock.MagicMock(return_value=mock_mapper)
        mock_mapper_function.__name__ = "mock_mapper_function"
        loaders._config.mapper_functions.register(mock_mapper_function)
        assert "mock_batches_function" in loaders._config.batches_from_mapper_functions
        assert "mock_mapper_function" in loaders._config.mapper_functions
        data_path = "test/data/path"
        variables = ["var1"]
        batches_kwargs = {"arg1": "value1", "arg2": 2}
        mapper_kwargs = {"arg3": 3}
        config = loaders._config.BatchesFromMapperConfig(
            mapper_config=loaders._config.MapperConfig(
                data_path=data_path,
                mapper_function="mock_mapper_function",
                mapper_kwargs=mapper_kwargs,
            ),
            batches_function="mock_batches_function",
            batches_kwargs=batches_kwargs,
        )
        result = config.load_batches(variables=variables)
        assert result is mock_batches
        mock_batches_function.assert_called_once_with(
            mock_mapper, variables, **batches_kwargs
        )
        mock_mapper_function.assert_called_once_with(data_path, **mapper_kwargs)


def test_batches_from_mapper_load_mapper():
    with batches_from_mapper_context(), mapper_context():
        mock_batches = [xr.Dataset()]
        mock_batches_function = unittest.mock.MagicMock(return_value=mock_batches)
        mock_batches_function.__name__ = "mock_batches_function"
        loaders._config.batches_from_mapper_functions.register(mock_batches_function)
        mock_mapper = {"key": xr.Dataset()}
        mock_mapper_function = unittest.mock.MagicMock(return_value=mock_mapper)
        mock_mapper_function.__name__ = "mock_mapper_function"
        loaders._config.mapper_functions.register(mock_mapper_function)
        assert "mock_batches_function" in loaders._config.batches_from_mapper_functions
        assert "mock_mapper_function" in loaders._config.mapper_functions
        data_path = "test/data/path"
        batches_kwargs = {"arg1": "value1", "arg2": 2}
        mapper_kwargs = {"arg3": 3}
        config = loaders._config.BatchesFromMapperConfig(
            mapper_config=loaders._config.MapperConfig(
                data_path=data_path,
                mapper_function="mock_mapper_function",
                mapper_kwargs=mapper_kwargs,
            ),
            batches_function="mock_batches_function",
            batches_kwargs=batches_kwargs,
        )
        result = config.load_mapper()
        assert result is mock_mapper
        mock_batches_function.assert_not_called
        mock_mapper_function.assert_called_once_with(data_path, **mapper_kwargs)


def test_load_batches_from_mapper_raises_if_registered_with_wrong_decorator():
    with batches_from_mapper_context(), mapper_context():
        mock_batches = [xr.Dataset()]
        mock_batches_function = unittest.mock.MagicMock(return_value=mock_batches)
        mock_batches_function.__name__ = "mock_batches_function"
        loaders._config.batches_functions.register(mock_batches_function)
        mock_mapper = {"key": xr.Dataset()}
        mock_mapper_function = unittest.mock.MagicMock(return_value=mock_mapper)
        mock_mapper_function.__name__ = "mock_mapper_function"
        loaders._config.mapper_functions.register(mock_mapper_function)
        data_path = "test/data/path"
        variables = ["var1"]
        batches_kwargs = {"arg1": "value1", "arg2": 2}
        mapper_kwargs = {"arg3": 3}
        config = loaders._config.BatchesFromMapperConfig(
            mapper_config=loaders._config.MapperConfig(
                data_path=data_path,
                mapper_function="mock_mapper_function",
                mapper_kwargs=mapper_kwargs,
            ),
            batches_function="mock_batches_function",
            batches_kwargs=batches_kwargs,
        )
        with pytest.raises(KeyError):
            config.load_batches(variables=variables)


@pytest.mark.parametrize(
    "data, expected_class",
    [
        pytest.param(
            {
                "data_path": "mock/data/path",
                "batches_function": "mock_batches_function",
                "batches_kwargs": {},
            },
            loaders._config.BatchesConfig,
        ),
        pytest.param(
            {
                "mapper_config": {
                    "data_path": "mock/data/path",
                    "mapper_function": "mock_mapper_function",
                    "mapper_kwargs": {},
                },
                "batches_function": "mock_batches_function",
                "batches_kwargs": {},
            },
            loaders._config.BatchesFromMapperConfig,
        ),
    ],
)
def test_batches_loader_from_dict(data, expected_class):
    result = loaders._config.BatchesLoader.from_dict(data)
    assert type(result) is expected_class


def test_safe_dump_data_config():
    """
    Test that dataclass.asdict and pyyaml can be used to save BatchesConfig.
    """
    config = loaders.BatchesConfig(
        data_path="/my/path",
        batches_function="batches_func",
        batches_kwargs={"key": "value"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "config.yaml")
        with open(filename, "w") as f:
            as_dict = dataclasses.asdict(config)
            yaml.safe_dump(as_dict, f)
        from_dict = loaders.BatchesConfig(**as_dict)
        assert config == from_dict
