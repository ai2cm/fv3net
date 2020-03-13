import argparse
import yaml
import json
import os
import uuid
from typing import List, Mapping, Any, Hashable


def get_experiment_steps_and_args(config_file: str):
    """
    Load all arguments for orchestration script from config and
    dump in a JSON format to be consumed.
    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Resolve inputs, outputs, and other config parameters
    workflow_steps = config["experiment"]["steps_to_run"]
    all_steps_config = config["experiment"]["steps_config"]
    try:
        current_steps_config = {step: all_steps_config[step] for step in workflow_steps }
    except KeyError:
        raise('Workflow steps list contains a step not defined in steps config.')
    config["experiment"]["steps_config"] = current_steps_config
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

        for arg in args_config:
            if isinstance(args_config[arg], Mapping):
                source_info = args_config[arg]
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
                    previous_step = steps_config.get(from_key, None)
                    if previous_step is not None:
                        source_info["location"] = previous_step
                    else:
                        raise KeyError(
                            f"A step argument specified 'from' another step requires that "
                            f"the other step also be run as part of the same workflow. "
                            f"Add '{from_key}' to the workflow or specify '{arg}' with " 
                            f"'location' instead."
                        )
                else:
                    raise KeyError(
                        f"An arg of {step_name} is provided as a key-value pair,"
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


def _get_all_step_arguments(config: Mapping):
    """Get a dictionary of each step with i/o and methedological arguments"""

    steps_config = config["experiment"]["steps_config"]
    all_step_arguments = {}
    for step, step_config in steps_config.items():
        step_args = [step_config["command"]]
        required_args = []
        optional_args = []
        for key, value in step_config["args"].items():
            if key.startswith('--'):
                optional_args.extend([key, _resolve_arg_value(value)])
            else:
                required_args.append(_resolve_arg_value(value))
        output_location = step_config["output_location"]
        step_args.extend(required_args)
        step_args.append(output_location)
        step_args.extend(optional_args)
        all_step_arguments[step] = ' '.join(step_args)
    print(all_step_arguments)

    return all_step_arguments


def _generate_output_path_from_config(
    step_name: str, step_config: Mapping, max_config_stubs: int = 3
):
    """generate an output location stub from a step's argument configuration"""

    output_str = step_name
    arg_config = step_config.get("args", None)
    arg_strs = []
    n_stubs = 0
    for key in [key for key in arg_config if not isinstance(arg_config[key], Mapping)]:
        n_stubs =+ 1
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


# def _generate_args(step_config: Mapping):
#     """
#     Generate the arguments for the step as positional arguments
#     in a string followed by optional arguments.
#     """
#     arg_config = step_config.get("extra_args", None)

#     if arg_config is not None:
#         optional_args = []
#         required_args = []
#         for arg_key, arg_value in arg_config.items():
#             if arg_key[:2] == "--":
#                 optional_args += [arg_key, str(arg_value)]
#             else:
#                 required_args.append(str(arg_value))

#         combined_args = " ".join(required_args + optional_args)
#     else:
#         combined_args = ""

#     return combined_args

def _resolve_arg_value(value: Any) -> Hashable:
    if isinstance(value, Mapping):
        location_value = value.get("location", None)
        if location_value is None:
            raise ValueError("Argument 'location' value not specified.")
        else:
            return str(location_value) 
    else:
        return str(value)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Location of workflow config yaml."
    )

    args = parser.parse_args()

    exp_args = get_experiment_steps_and_args(args.config_file)

    # Print JSON to stdout to be consumed by shell script
    print(exp_args)
