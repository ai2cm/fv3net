import argparse
import yaml
import json
import os
import uuid


def get_experiment_args(args):
    """Load all arguments for orchestration script from config"""
    
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    experiment_config = config['experiment']
    workflow_steps = experiment_config['workflow_steps']
    all_step_arguments = _get_all_step_arguments(workflow_steps, config)

    return json.dumps(all_step_arguments)


def _get_experiment_path(config, unique_id=None):

    proto = config['storage_proto']
    root = config['storage_root']
    experiment = config['experiment']

    if proto == '' or proto is None:
        proto = 'file'

    experiment_name = f"{experiment['name']}"

    if unique_id is not None:
        experiment_name = f"{experiment_name}-{unique_id}"

    return f"{proto}://{root}/{experiment_name}"


def _get_all_step_arguments(workflow_steps, config):
    """Get a dictionary of each step with i/o and methedological arguments"""
    
    if config['experiment']['unique_id']:
        unique_id = str(uuid.uuid4())[-8:]
    else:
        unique_id = None
    
    root_experiment_path = _get_experiment_path(config, unique_id=unique_id)
    steps_config = config["experiment"]["steps"]
    all_step_arguments = {}
    for i, step in enumerate(workflow_steps):
        curr_config = steps_config[step]
        input_location = _get_input_location(step, curr_config, steps_config, root_experiment_path)
        output_location = _get_output_location(step, curr_config, root_experiment_path)
        method_args = _generate_method_args(curr_config, unique_id=unique_id)
        step_args = " ".join([input_location, output_location, method_args])
        all_step_arguments[step] = step_args

    return all_step_arguments


def _get_input_location(step, step_config, all_steps_config, root_path):
    """get an individual steps input location"""
    
    if "input_location" in step_config:
        # User-defined
        input_location = step_config["input_location"]
    elif "input_from" in step_config:
        # Auto-generated
        prev_step = step_config["input_from"]
        prev_config = all_steps_config[prev_step]
        input_location = _get_output_location(prev_step, prev_config, root_path)
    else:
        raise KeyError(f"Missing specification of input location for step: {step}")

    return input_location


def _get_output_location(step_name, step_config, root_path):
    """Get a steps output location or generate if not specified"""
    if "output_location" in step_config:
        # User-defined
        output_location = step_config["output_location"]
    else:
        # Auto-generated
        output_stub = _generate_output_path_from_config(step_name, step_config)
        output_location = os.path.join(root_path, output_stub)

    return output_location
    

def _generate_output_path_from_config(step_name, step_config, max_config_stubs=3):
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


def _generate_method_args(step_config, unique_id=None):
    """generate the methodlogical arguments for the step"""
    method_config = step_config.get("method", None)
    methods_to_add_unique_id = step_config.get("add_unique_id", [])

    if method_config is not None:
        method_args = []
        for key, val in method_config.items():
            if key in methods_to_add_unique_id and unique_id is not None:
                val = f"{val}-{unique_id}"
            else:
                val = str(val)
            method_args.append(val)
        combined_method_args = " ".join(method_args)
    else:
        combined_method_args = ""

    return combined_method_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Location of workflow config yaml."
    )
    
    args = parser.parse_args()
    
    # run the function
    exp_args = get_experiment_args(args)
    print(exp_args)
