import pytest
import unittest.mock
import loaders
import xarray as xr
import tempfile
import os
import yaml
import dataclasses
from loaders.testing import mapper_context, batches_context, batches_from_mapper_context


def test_load_mapper():
    with mapper_context():
        mock_mapper = {"key": xr.Dataset()}
        mock_mapper_function = unittest.mock.MagicMock(return_value=mock_mapper)
        mock_mapper_function.__name__ = "mock_mapper_function"
        loaders._config.mapper_functions.register(mock_mapper_function)
        assert "mock_mapper_function" in loaders._config.mapper_functions
        data_path = "test/data/path"
        mapper_kwargs = {"arg1": "value1", "arg2": 2, "data_path": data_path}
        config = loaders._config.MapperConfig(
            function="mock_mapper_function", kwargs=mapper_kwargs,
        )
        result = config.load_mapper()
        assert result is mock_mapper
        mock_mapper_function.assert_called_once_with(**mapper_kwargs)


def test_mapper_config_raises_on_invalid_mapper_function():
    with mapper_context():
        with pytest.raises(ValueError):
            loaders.MapperConfig(function="missing_function", kwargs={})


def test_batches_config_raises_on_invalid_batches_function():
    with batches_context():
        with pytest.raises(ValueError):
            loaders.BatchesConfig(function="missing_function", kwargs={})


def test_batches_from_mapper_config_raises_on_invalid_batches_function():
    with batches_from_mapper_context():
        with pytest.raises(ValueError):
            loaders.BatchesFromMapperConfig(
                mapper_config=loaders.MapperConfig(
                    function="open_zarr", kwargs={"data_path": "/"}
                ),
                function="missing_function",
                kwargs={},
            )


def test_expected_mapper_functions_exist():
    # this test exists to ensure we don't accidentally remove functions we
    # currently use in configs, if you are deleting an option we no longer use
    # you can delete it here
    expected_functions = (
        "open_nudge_to_obs",
        "open_nudge_to_fine",
        "open_high_res_diags",
        "open_fine_resolution_nudging_hybrid",
        "open_precomputed_fine_resolution_nudging_hybrid",
        "open_fine_resolution",
        "open_precomputed_fine_resolution",
        "open_nudge_to_fine_multiple_datasets",
        "open_zarr",
    )
    for expected in expected_functions:
        assert expected in loaders._config.mapper_functions
    missing_expected = list(
        set(loaders._config.mapper_functions).difference(expected_functions)
    )
    assert len(missing_expected) == 0, f"add {missing_expected} to expected_functions"


def test_expected_batches_functions_exist():
    # this test exists to ensure we don't accidentally remove functions we
    # currently use in configs, if you are deleting an option we no longer use
    # you can delete it here
    expected_functions = (
        "batches_from_serialized",
        "batches_from_netcdf",
    )
    for expected in expected_functions:
        assert expected in loaders._config.batches_functions
    missing_expected = list(
        set(loaders._config.batches_functions).difference(expected_functions)
    )
    assert len(missing_expected) == 0, f"add {missing_expected} to expected_functions"


def test_expected_batches_from_mapper_functions_exist():
    # this test exists to ensure we don't accidentally remove functions we
    # currently use in configs, if you are deleting an option we no longer use
    # you can delete it here
    expected_functions = ("batches_from_mapper",)
    for expected in expected_functions:
        assert expected in loaders._config.batches_from_mapper_functions
    missing_expected = list(
        set(loaders._config.batches_from_mapper_functions).difference(
            expected_functions
        )
    )
    assert len(missing_expected) == 0, f"add {missing_expected} to expected_functions"


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
    with batches_context() as mock_batches_function:
        data_path = "test/data/path"
        variables = ["var1"]
        batches_kwargs = {"arg1": "value1", "arg2": 2, "data_path": data_path}
        config = loaders._config.BatchesConfig(
            function="mock_batches_function", kwargs=batches_kwargs,
        )
        result = config.load_batches(variables=variables)
        assert result is mock_batches_function.return_value
        mock_batches_function.assert_called_once_with(
            variable_names=variables, **batches_kwargs
        )


def batches_from_mapper_init():
    data_path = "test/data/path"
    variables = ["var1"]
    batches_kwargs = {"arg1": "value1", "arg2": 2}
    mapper_kwargs = {"arg3": 3, "data_path": data_path}
    config = loaders._config.BatchesFromMapperConfig(
        mapper_config=loaders._config.MapperConfig(
            function="mock_mapper_function", kwargs=mapper_kwargs,
        ),
        function="mock_batches_function",
        kwargs=batches_kwargs,
    )
    return data_path, variables, batches_kwargs, mapper_kwargs, config


def test_load_batches_from_mapper():
    with batches_from_mapper_context() as mock_batches_function, mapper_context() as mock_mapper_function:  # noqa: E501
        (
            data_path,
            variables,
            batches_kwargs,
            mapper_kwargs,
            config,
        ) = batches_from_mapper_init()
        result = config.load_batches(variables=variables)
        assert result is mock_batches_function.return_value
        mock_batches_function.assert_called_once_with(
            mock_mapper_function.return_value,
            variable_names=variables,
            **batches_kwargs,
        )
        mock_mapper_function.assert_called_once_with(**mapper_kwargs)


def test_batches_from_mapper_load_mapper():
    with batches_from_mapper_context() as mock_batches_function, mapper_context() as mock_mapper_function:  # noqa: E501
        data_path, _, _, mapper_kwargs, config = batches_from_mapper_init()
        result = config.load_mapper()
        assert result is mock_mapper_function.return_value
        mock_batches_function.assert_not_called
        mock_mapper_function.assert_called_once_with(**mapper_kwargs)


def test_load_batches_from_mapper_raises_if_registered_with_wrong_decorator():
    with batches_from_mapper_context(), mapper_context():
        mock_batches = [xr.Dataset()]
        another_mock_batches_function = unittest.mock.MagicMock(
            return_value=mock_batches
        )
        another_mock_batches_function.__name__ = "another_mock_batches_function"
        loaders._config.batches_functions.register(another_mock_batches_function)
        data_path = "test/data/path"
        batches_kwargs = {"arg1": "value1", "arg2": 2}
        mapper_kwargs = {"arg3": 3, "data_path": data_path}
        with pytest.raises(ValueError):
            loaders._config.BatchesFromMapperConfig(
                mapper_config=loaders._config.MapperConfig(
                    function="mock_mapper_function", kwargs=mapper_kwargs,
                ),
                function="another_mock_batches_function",
                kwargs=batches_kwargs,
            )


@pytest.mark.parametrize(
    "data, expected_class",
    [
        pytest.param(
            {
                "mapper_config": {
                    "function": "open_zarr",
                    "kwargs": {"data_path": "mock/data/path"},
                },
                "function": "batches_from_mapper",
                "kwargs": {},
            },
            loaders._config.BatchesFromMapperConfig,
            id="batches_from_mapper_config",
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
        function="batches_from_mapper", kwargs={"timesteps": ["1", "2", "2"]},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "config.yaml")
        with open(filename, "w") as f:
            as_dict = dataclasses.asdict(config)
            yaml.safe_dump(as_dict, f)
        from_dict = loaders.BatchesConfig(**as_dict)
        assert config == from_dict


def test_duplicate_times_raise_error_in_batches_from_mapper():
    data_config = {
        "mapper_config": {"function": "open_zarr", "kwargs": {}},
        "function": "batches_from_mapper",
        "kwargs": {"timesteps": ["1", "2", "2"]},
    }
    with pytest.raises(ValueError):
        loaders._config.BatchesFromMapperConfig.from_dict(data_config)


@pytest.mark.parametrize(
    "variables_config, variables_arg",
    [
        pytest.param([], [], id="no_variables"),
        pytest.param(["a", "b"], [], id="config_variables"),
        pytest.param([], ["a", "b"], id="arg_variables"),
        pytest.param(["a", "b"], ["c"], id="different_variables"),
        pytest.param(["a", "b"], ["b", "c"], id="overlapping_variables"),
    ],
)
def test_batches_from_mapper_combines_variables(variables_config, variables_arg):
    result = None
    expected_return_value = unittest.mock.MagicMock()
    with mapper_context(), batches_from_mapper_context():

        @loaders._config.mapper_functions.register
        def mock_mapper():
            return None

        @loaders._config.batches_from_mapper_functions.register
        def mock_batches_from_mapper(
            mapping_function, variable_names, mapping_kwargs=None,
        ):
            nonlocal result
            result = variable_names
            return expected_return_value

        loader = loaders._config.BatchesLoader.from_dict(
            {
                "function": "mock_batches_from_mapper",
                "mapper_config": loaders._config.MapperConfig(
                    function="mock_mapper", kwargs={}
                ),
                "kwargs": {"variable_names": variables_config},
            }
        )
        return_value = loader.load_batches(variables_arg)
    assert return_value == expected_return_value
    assert set(result) == set(variables_config).union(variables_arg)


@pytest.mark.parametrize(
    "variables_config, variables_arg",
    [
        pytest.param([], [], id="no_variables"),
        pytest.param(["a", "b"], [], id="config_variables"),
        pytest.param([], ["a", "b"], id="arg_variables"),
        pytest.param(["a", "b"], ["c"], id="different_variables"),
        pytest.param(["a", "b"], ["b", "c"], id="overlapping_variables"),
    ],
)
def test_batches_combines_variables(variables_config, variables_arg):
    result = None
    expected_return_value = unittest.mock.MagicMock()
    with batches_context():

        @loaders._config.batches_functions.register
        def mock_batches(variable_names,):
            nonlocal result
            result = variable_names
            return expected_return_value

        loader = loaders._config.BatchesLoader.from_dict(
            {
                "function": "mock_batches",
                "kwargs": {"variable_names": variables_config},
            }
        )
        return_value = loader.load_batches(variables_arg)
    assert return_value == expected_return_value
    assert set(result) == set(variables_config).union(variables_arg)
