import pytest

from get_experiment_steps_and_args import (
    _get_all_step_arguments,
    _resolve_command,
    _resolve_arg_values,
    _resolve_arg_values_argo,
)


@pytest.fixture()
def test_config():
    return {
        "experiment": {
            "steps_config": {
                "argo_step": {
                    "argo": "argo submit some.yaml",
                    "args": {
                        "--optional_arg": "optional_arg_value",
                        "required_arg": "required_arg_value",
                        "required_location_arg": {"location": "data_location"},
                    },
                    "output_location": "output_location",
                },
                "python_step": {
                    "args": {
                        "--optional_arg": "optional_arg_value",
                        "required_arg": "required_arg_value",
                        "required_location_arg": {"location": "data_location"},
                    },
                    "command": "python -m test_script",
                    "output_location": "output_location",
                },
            }
        }
    }


def test__get_all_step_arguments(test_config):
    all_commands = _get_all_step_arguments(test_config)
    assert set(all_commands.keys()) == set(
        test_config["experiment"]["steps_config"].keys()
    )
    assert (
        all_commands["argo_step"]
        == "argo submit some.yaml -p required_arg=required_arg_value -p required_location_arg=data_location -p output_location=output_location --optional_arg optional_arg_value"
    )
    assert (
        all_commands["python_step"]
        == "python -m test_script required_arg_value data_location output_location --optional_arg optional_arg_value"
    )


def test__resolve_command(test_config):
    assert (
        _resolve_command(test_config["experiment"]["steps_config"]["argo_step"])
        == "argo"
    )
    assert (
        _resolve_command(test_config["experiment"]["steps_config"]["python_step"])
        == "command"
    )


def test_resolve_arg_values():
    assert _resolve_arg_values("required_arg", 10) == "10"
    assert _resolve_arg_values("--optional_arg", 10) == "--optional_arg 10"


def test_resolve_arg_values_argo():
    assert _resolve_arg_values_argo("required_arg", 10) == "-p required_arg=10"
    assert _resolve_arg_values_argo("--optional_arg", 10) == "--optional_arg 10"
