import argparse
import yaml
import json
import os
import uuid
from typing import List


def get_experiment_steps_and_args(config_file: str):
    """
    Load all arguments for orchestration script from config and
    dump in a JSON format to be consumed.
    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Resolve inputs, outputs, and other config parameters
    _apply_config_transforms(config)
    workflow_steps_config = config["experiment"]["workflow_steps"]
    all_step_commands, all_step_arguments = _get_all_step_arguments(
        workflow_steps_config, config
    )
    experiment_steps_and_args = {
        "name": config["experiment"]["name"],
        "workflow": " ".join([step for step in workflow_steps_config]),
        "commands": all_step_commands,
        "arguments": all_step_arguments,
    }
    return json.dumps(experiment_steps_and_args)


def _apply_config_transforms(config: dict):
    """
    Transforms to apply to the configuration dictionary.  All transforms
    are assumed to be in-place.
    """

    _add_unique_id(config)
    _use_top_level_var(config)
    _resolve_output_location(config)
    _resolve_input_from(config)


def _add_unique_id(config: dict):
    """Add a shared uuid to specified configuration paramters"""

    exp_config = config["experiment"]
    if not exp_config["unique_id"]:
        return

    unique_id = str(uuid.uuid4())[-8:]
    exp_name = exp_config["name"]
    exp_config["name"] = exp_name + f"-{unique_id}"

    # Go through all step configuration to check for method specific uuid additions
    for step_name, step_config in exp_config["steps"].items():
        config_transforms = step_config.get("config_transforms", None)

        if config_transforms and "add_unique_id" in config_transforms:
            vars_to_add_unique = step_config["config_transforms"]["add_unique_id"]
            for config_var in vars_to_add_unique:
                method_val = step_config["method"][config_var]
                method_val += f"-{unique_id}"
                step_config["method"][config_var] = method_val


def _resolve_output_location(config: dict):
    """Get the step output location if one is not specified"""
    root_exp_path = _get_experiment_path(config)
    all_steps_config = config["experiment"]["steps"]

    for step_name, step_config in all_steps_config.items():

        if "output_location" in step_config:
            continue
        else:
            output_stub = _generate_output_path_from_config(step_name, step_config)
            location = os.path.join(root_exp_path, output_stub)
            step_config["output_location"] = location


def _resolve_input_from(config: dict):
    """
    Get the step input location if not specified.  Derives from previous
    steps if the "from" keyword is used along with a step name.
    """

    all_steps_config = config["experiment"]["steps"]

    for step_name, step_config in all_steps_config.items():
        input_config = step_config["inputs"]

        for input_source, source_info in input_config.items():
            location = source_info.get("location", None)
            from_key = source_info.get("from", None)

            if location is not None and from_key is not None:
                raise ValueError(
                    f"Ambiguous input location for {step_name}-{input_source}."
                    f" Both 'from' and 'location' were specified"
                )
            if location is not None:
                continue
            elif from_key is not None:
                source_info["location"] = all_steps_config[from_key]["output_location"]
            else:
                raise KeyError(
                    f"Input section of {step_name} should have either 'location' "
                    "or 'from' specified in the orchestration configuration"
                )


def _use_top_level_var(config: dict):
    """
    Interpolate parameters marked to use a top-level variable from the YAML file.
    """

    experiment_vars = config["experiment"]["experiment_vars"]
    all_steps_config = config["experiment"]["steps"]

    for step_name, step_config in all_steps_config.items():

        if "config_transforms" not in step_config:
            continue
        elif "use_top_level" not in step_config["config_transforms"]:
            continue

        vars_to_transform = step_config["config_transforms"]["use_top_level"]

        for varname in vars_to_transform:

            exp_var_key = step_config["method"][varname]
            step_config["method"][varname] = experiment_vars[exp_var_key]


def _get_experiment_path(config: dict):
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


def _get_all_step_arguments(workflow_steps: List[str], config: dict):
    """Get a dictionary of each step with i/o and methedological arguments"""

    steps_config = config["experiment"]["steps"]
    all_step_commands = {}
    all_step_arguments = {}
    for i, step in enumerate(workflow_steps):
        curr_config = steps_config[step]
        all_input_locations = [
            input_info["location"] for input_info in curr_config["inputs"].values()
        ]
        output_location = curr_config["output_location"]
        method_args = _generate_method_args(curr_config)

        input_args = " ".join(all_input_locations)
        step_args = " ".join([input_args, output_location, method_args])
        all_step_arguments[step] = step_args
        all_step_commands[step] = curr_config["command"]

    return all_step_commands, all_step_arguments


def _generate_output_path_from_config(
    step_name: str, step_config: dict, max_config_stubs: int = 3
):
    """generate an output location stub from a step's methodological config"""

    output_str = step_name
    method_config = step_config.get("method", None)
    if method_config is not None:
        method_strs = []
        for i, (key, val) in enumerate(method_config.items()):
            if i >= max_config_stubs:
                break

            val = str(val)

            # get last part of path so string isn't so long
            if "/" in val:
                val = val.split("/")[-1]

            key_val = f"{key}_{val}"
            method_strs.append(key_val)
        method_output_stub = "_".join(method_strs)
        output_str += "_" + method_output_stub

    return output_str


def _generate_method_args(step_config: dict):
    """
    Generate the methodological arguments for the step as positional arguments
    in a string.
    """
    method_config = step_config.get("method", None)

    if method_config is not None:
        method_args = [str(val) for val in method_config.values()]
        combined_method_args = " ".join(method_args)
    else:
        combined_method_args = ""

    return combined_method_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Location of workflow config yaml."
    )

    args = parser.parse_args()

    exp_args = get_experiment_steps_and_args(args.config_file)

    # Print JSON to stdout to be consumed by shell script
    print(exp_args)
