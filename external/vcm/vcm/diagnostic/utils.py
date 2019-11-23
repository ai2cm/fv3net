from dataclasses import dataclass
from typing import List
import ufuncs
import yaml



VERTICAL_GRID_VAR = 'pfull'
FUNCTION_MAP = {
    'mean': ufuncs.mean,
    'test_func': ufuncs.test_func
}


@dataclass
class PlotConfig:
    var: str
    plot_name: str
    plot_type: str
    dim_slices: List[dict]
    functions: List
    function_kwargs: List[dict]


def load_ufuncs(raw_config):
    functions, kwargs = [], []
    for function_spec in raw_config['function_specifications']:
        # sacrificing some ugliness here so that specification in the yaml is safer / more readable
        # single entry dict seemed like a more foolproof format than list of func name and kwargs
        function_name, function_kwargs = list(function_spec.keys())[0], list(function_spec.values())[0]
        if function_name not in FUNCTION_MAP:
            raise ValueError("Function name {} is not in the function map.")
        functions.append(FUNCTION_MAP[function_name])
        kwargs.append(function_kwargs)
    return functions, kwargs


def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            raw_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError("Bad yaml config: {}".format(exc))
    plot_configs = []
    for raw_config in raw_configs:
        dim_slices = [
            {dim: slice(indices[0], indices[1])}
                for dim, indices in raw_config['dim_slices'].items()
        ]
        functions, kwargs = load_ufuncs(raw_config)
        plot_config = PlotConfig(
            plot_name=raw_config['plot_name'],
            plot_type=raw_config['plot_type'],
            var=raw_config['variable'],
            dim_slices=dim_slices,
            functions=functions,
            kwargs=kwargs
        )
        plot_configs.append(plot_config)
    return plot_configs



