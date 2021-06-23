import pytest
import unittest.mock
import loaders._config
import contextlib
import xarray as xr
import tempfile
import os
import yaml
import dataclasses


@contextlib.contextmanager
def mapper_context():
    original_functions = {**loaders._config.MAPPER_FUNCTIONS}
    loaders._config.MAPPER_FUNCTIONS.clear()
    yield
    loaders._config.MAPPER_FUNCTIONS.clear()
    loaders._config.MAPPER_FUNCTIONS.update(original_functions)


@contextlib.contextmanager
def batches_context():
    original_functions = {**loaders._config.BATCHES_FUNCTIONS}
    loaders._config.BATCHES_FUNCTIONS.clear()
    yield
    loaders._config.BATCHES_FUNCTIONS.clear()
    loaders._config.BATCHES_FUNCTIONS.update(original_functions)


def test_load_mapper():
    with mapper_context():
        mock_mapper = {"key": xr.Dataset()}
        mock_mapper_function = unittest.mock.MagicMock(return_value=mock_mapper)
        mock_mapper_function.__name__ = "mock_mapper_function"
        loaders._config.register_mapper_function(mock_mapper_function)
        assert "mock_mapper_function" in loaders._config.MAPPER_FUNCTIONS
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
        assert expected in loaders._config.MAPPER_FUNCTIONS


def test_expected_batches_functions_exist():
    # this test exists to ensure we don't accidentally remove functions we
    # currently use in configs, if you are deleting an option we no longer use
    # you can delete it here
    for expected in (
        "batches_from_geodata",
        "batches_from_serialized",
        "diagnostic_batches_from_geodata",
    ):
        assert expected in loaders._config.BATCHES_FUNCTIONS


def test_load_batches():
    with batches_context():
        mock_batches = [xr.Dataset()]
        mock_batches_function = unittest.mock.MagicMock(return_value=mock_batches)
        mock_batches_function.__name__ = "mock_batches_function"
        loaders._config.register_batches_function(mock_batches_function)
        assert "mock_batches_function" in loaders._config.BATCHES_FUNCTIONS
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
    with batches_context(), mapper_context():
        mock_batches = [xr.Dataset()]
        mock_batches_function = unittest.mock.MagicMock(return_value=mock_batches)
        mock_batches_function.__name__ = "mock_batches_function"
        loaders._config.register_batches_function(mock_batches_function)
        mock_mapper = {"key": xr.Dataset()}
        mock_mapper_function = unittest.mock.MagicMock(return_value=mock_mapper)
        mock_mapper_function.__name__ = "mock_mapper_function"
        loaders._config.register_mapper_function(mock_mapper_function)
        assert "mock_batches_function" in loaders._config.BATCHES_FUNCTIONS
        assert "mock_mapper_function" in loaders._config.MAPPER_FUNCTIONS
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


@pytest.mark.parametrize(
    "data, expected_class",
    [
        pytest.param(
            {
                "data_path": "mock/data/path",
                "batches_function": "mock_batch_function",
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
                "batches_function": "mock_batch_function",
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
        batch_function="batch_func",
        batch_kwargs={"key": "value"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "config.yaml")
        with open(filename, "w") as f:
            as_dict = dataclasses.asdict(config)
            yaml.safe_dump(as_dict, f)
        from_dict = loaders.BatchesConfig(**as_dict)
        assert config == from_dict
