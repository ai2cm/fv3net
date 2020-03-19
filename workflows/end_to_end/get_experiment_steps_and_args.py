 import argparse
import yaml
import json
import os
import uuid
from typing import List, Mapping, Any, Hashable
from dataflow import COARSEN_RESTARTS_DATAFLOW_ARGS, CREATE_TRAINING_DATAFLOW_ARGS

DATAFLOW_ARGS_MAPPING = {
    "coarsen_restarts": COARSEN_RESTARTS_DATAFLOW_ARGS,
    "create_training_data": CREATE_TRAINING_DATAFLOW_ARGS,
}


def get_experiment_steps_and_args(config_file: str):
    """
    Load all arguments for orchestration script from config and
    dump in a JSON format to be consumed.
    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Resolve inputs, outputs, and other config parameters
    workflow_steps = config["experiment"]["steps_to_run"]
    if any(
        [step not in config["experiment"]["steps_config"] for step in workflow_steps]
    ):
        raise KeyError(
            "'steps_to_run' list contains a step not defined in 'steps_config'."
        )
    _apply_config_transforms(config)
    all_step_arguments = _get_all_step_arguments(config)
    experiment_steps_and_args = {
        "name": config["experiment"]["name"],
        "workflow": " ".join([step for step in workflow_steps]),
        "command_and_args": all_step_arguments,
    }
    return json.dumps(experiment_steps_and_args)


def _apply_config_transforms(config: Mapping):
    """
    Transforms to apply to the configuration dictionary.  All transforms
    are assumed to be in-place.
    """
    _add_unique_id(config)
    _resolve_output_location(config)
    _resolve_input_from(config)
    _resolve_dataflow_args(config)


def _add_unique_id(config: Mapping):
    """Add a shared uuid to specified configuration paramters"""

    exp_config = config["experiment"]
    if not exp_config["unique_id"]:
        return

    unique_id = str(uuid.uuid4())[-8:]
    exp_name = exp_config["name"]
    exp_config["name"] = exp_name + f"-{unique_id}"


def _resolve_output_location(config: Mapping):
    """Get the step output location if one is not specified"""
    root_exp_path = _get_experiment_path(config)
    steps_config = config["experiment"]["steps_config"]

    for step_name, step_config in steps_config.items():

        if "output_location" in step_config:
            continue
        else:
            output_stub = _generate_output_path_from_config(step_name, step_config)
            location = os.path.join(root_exp_path, output_stub)
            step_config["output_location"] = location


def _resolve_input_from(config: Mapping):
    """
    Get the step input location if not specified.  Derives from previous
    steps if the "from" keyword is used along with a step name.
    """

    steps_config = config["experiment"]["steps_config"]

    for step_name, step_config in steps_config.items():
        args_config = step_config["args"]

        for arg, val in args_config.items():
            if isinstance(val, Mapping):
                _resolve_input_mapping(val, steps_config, arg)


def _resolve_input_mapping(input_mapping: Mapping, steps_config: Mapping, arg: str):

    location = input_mapping.get("location", None)
    from_key = input_mapping.get("from", None)

    if location is not None and from_key is not None:
        raise ValueError(
            f"Ambiguous input location for {arg}."
            f" Both 'from' and 'location' were specified"
        )
    if location is not None:
        return
    elif from_key is not None:
        previous_step = steps_config.get(from_key, None)
        if previous_step is not None:
            input_mapping["location"] = previous_step["output_location"]
        else:
            raise KeyError(
                f"A step argument specified 'from' another step requires "
                f"that the other step's cofiguration be specified. Add "
                f"'{from_key}' to the configuration or specify '{arg}' "
                f"with 'location' instead."
            )
    else:
        raise KeyError(
            f"{arg} is provided as a key-value pair,"
            f" but only 'location' or 'from' may be specified."
        )


def _get_experiment_path(config: Mapping):
    """Get root directory path for experiment output."""

    proto = config["storage_proto"]
    root = config["storage_root"]
    experiment = config["experiment"]

    if proto == "" or proto is None:
        proto = "file"

    if proto != "file" and proto != "gs":
        raise ValueError(
            f"Protocol, {proto}, is not currently supported. Please use "
            f"'file' or 'gs'"
        )

    experiment_name = f"{experiment['name']}"

    return f"{proto}://{root}/{experiment_name}"


def _resolve_dataflow_args(config: Mapping):
    """Add dataflow arguments to step if it is the job runner"""

    steps_config = config["experiment"]["steps_config"]
    for step, step_config in steps_config.items():
        dataflow_arg = step_config["args"].get("--runner", None)
        if dataflow_arg == "DataflowRunner":
            step_config["args"].update(DATAFLOW_ARGS_MAPPING[step])
        elif dataflow_arg == "DirectRunner":
            continue
        elif dataflow_arg is not None:
            raise ValueError(
                f"'runner' arg must be 'DataflowRunner' or 'DirectRunner'; "
                f"instead received '{dataflow_arg}'."
            )


def _get_all_step_arguments(config: Mapping):
    """Get a dictionary of each step with i/o and methedological arguments"""

    steps_config = config["experiment"]["steps_config"]
    all_step_arguments = {}
    for step, step_config in steps_config.items():
        step_args = [step_config["command"]]
        required_args = []
        optional_args = []
        for key, value in step_config["args"].items():
            arg_string = _resolve_arg_values(key, value)
            if arg_string.startswith("--"):
                optional_args.append(arg_string)
            else:
                required_args.append(arg_string)
        output_location = step_config["output_location"]
        step_args.extend(required_args)
        step_args.append(output_location)
        step_args.extend(optional_args)
        all_step_arguments[step] = " ".join(step_args)

    return all_step_arguments


def _generate_output_path_from_config(
    step_name: str, step_config: Mapping, max_config_stubs: int = 3
):
    """generate an output location stub from a step's argument configuration"""

    output_str = step_name
    arg_config = step_config.get("args", None)
    arg_strs = []
    non_map_args = {
        key: val for key, val in arg_config.items() if not isinstance(val, Mapping)
    }
    for n_stubs, (key, val) in enumerate(non_map_args.items(), 1):
        if n_stubs > max_config_stubs:
            break
        val = str(arg_config[key])

        # get last part of path so string isn't so long
        if "/" in val:
            val = val.split("/")[-1]

        key = key.strip("--")  # remove prefix of optional argument
        key_val = f"{key}_{val}"
        arg_strs.append(key_val)
    arg_output_stub = "_".join(arg_strs)
    output_str += "_" + arg_output_stub

    return output_str


def _resolve_arg_values(key: Hashable, value: Any) -> Hashable:
    """take a step args key-value pair and process into an appropriate arg string"
    """
    if isinstance(value, Mapping):
        # case for when the arg is a dict {"location" : path}
        location_value = value.get("location", None)
        if location_value is None:
            raise ValueError("Argument 'location' value not specified.")
        else:
            if key.startswith("--"):
                arg_values = " ".join([key, str(location_value)])
            else:
                arg_values = str(location_value)
    elif isinstance(value, List):
        # case for when the arg is a list
        # i.e., multiple optional args with same key, needed for dataflow packages
        multiple_optional_args = []
        for item in value:
            multiple_optional_args.extend([key, item])
        arg_values = " ".join(multiple_optional_args)
    else:
        if key.startswith("--"):
            arg_values = " ".join([key, str(value)])
        else:
            arg_values = str(value)
    return arg_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Location of workflow config yaml."
    )

    args = parser.parse_args()

    exp_args = get_experiment_steps_and_args(args.config_file)

    # Print JSON to stdout to be consumed by shell script
    print(exp_args)
