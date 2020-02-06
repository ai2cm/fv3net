import yaml
import fsspec
import os
import uuid


# STEP_CONFIG_VARS = ['input_location', 'output_location', 'method']


def create_experiment_path(args):
    config_location = args.workflow_config
    proto, root, experiment = _get_experiment_config(config_location)
    if proto == '' or proto is None:
        proto = 'file'
    experiment_name = f"{experiment['name']}_{str(uuid.uuid4())[-7:]}"
    print(f"{proto}://{root}/{experiment_name}")


def get_step_args(args):
    config_location = args.workflow_config
    experiment_path = args.experiment_path
    step = args.workflow_step
    _, _, experiment = _get_experiment_config(config_location)
    step_input = _get_step_input(experiment_path, step, experiment['steps'])
    step_output = _get_step_output(experiment_path, step, experiment['steps'])
    step_method = experiment['steps'][step]['method']
    print(step_input, step_output, step_method)
    
    
def _get_step_input(experiment_path, step, steps_config):
    previous_step, previous_step_method = _get_previous_step(step, steps_config)
    if step != 'coarsen':
        return os.path.join(experiment_path, f"{previous_step}_{previous_step_method}")
    else:
        return steps_config[previous_step]['output_location']

    
def _get_previous_step(step, steps_config):
    for other_step in steps_config:
        if other_step == steps_config[step]['input_method']:
            return other_step, steps_config[other_step]['method']
        

def _get_step_output(experiment_path, step, steps_config):
    return os.path.join(experiment_path, f"{step}_{steps_config[step]['method']}")


def _get_step_method(experiment_path, step, steps_config):
    return os.path.join(experiment_path, f"{step}_{steps_config[step]['method']}")
    
    
def _get_experiment_config(config_location):
    with open(config_location) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    proto = config['storage_proto']
    root = config['storage_root']
    experiment = config['experiment']
    return proto, root, experiment


# def _create_experiment_directories(proto, root, experiment_list):
#     fs = fsspec.filesystem(proto)
#     for experiment in experiment_list:
#         experiment = experiment['experiment']
#         experiment_name = f"{experiment['name']}_{str(uuid.uuid4())[-7:]}"
#         experiment_prefix = os.path.join(root, experiment_name)
#         print(experiment_prefix)
#         fs.makedirs(experiment_prefix, exist_ok = True)
#         _dict_to_nested_dirs(fs, experiment_prefix, experiment['steps'])
        
    
# def _dict_to_nested_dirs(fs, root, nested_dict):
#     for key in nested_dict:
#         if isinstance(nested_dict[key], dict) and _is_new_step(nested_dict[key]):
#             prefix = os.path.join(root, key)
#             fs.mkdir(prefix)
#             _dict_to_nested_dirs(fs, prefix, nested_dict[key])
            

# def _is_new_step(step_dict):
#     if step_dict['output_location'] == '.':
#         return True
    
    
# def _set_experiment_config(name, experiment):
#     for step in experiment['experiment']['steps']:
#         for var in STEP_CONFIG_VARS:
#             os.environment[f"{name}_{step}_{var}"] = step[var]
        
