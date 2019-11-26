from dataclasses import dataclass
import gcsfs
from typing import List
import xarray as xr
import yaml

from vcm.diagnostic import ufuncs

FUNCTION_MAP = {
    'mean_over_dim': ufuncs.mean_over_dim,
    'sum_over_dim': ufuncs.sum_over_dim,
}


@dataclass
class PlotConfig:
    diagnostic_variable: str
    plot_name: str
    plot_type: str
    dim_slices: dict
    functions: List
    function_kwargs: List[dict]


def read_zarr_from_gcs(gcs_url, project='vcm-ml'):
    fs = gcsfs.GCSFileSystem(project=project)
    return xr.open_zarr(fs.get_mapper(gcs_url))


def load_ufuncs(raw_config):

    if 'function_specifications' in raw_config:
        functions, kwargs = [], []
        for function_spec in raw_config['function_specifications']:
            # sacrificing some ugliness here so that specification in the yaml is safer / more readable:
            # single entry dict seemed like a more foolproof format than list of func name and kwargs
            function_name, function_kwargs = list(function_spec.keys())[0], list(function_spec.values())[0]
            if function_name not in FUNCTION_MAP:
                raise ValueError("Function name {} is not in the function map.".format(function_name))
            functions.append(FUNCTION_MAP[function_name])
            kwargs.append(function_kwargs)
        return functions, kwargs
    else:
        return [], []


def load_dim_slices(raw_config):
    dim_selection = {}

    if 'dim_slices' in raw_config:
        for dim, indices in raw_config['dim_slices'].items():
            if len(indices)==1:
                dim_selection[dim] = indices[0]
            else:
                dim_selection[dim] = slice(indices[0], indices[1])
        return dim_selection
    else:
        return {}


def load_configs(config_path):
    with open(config_path, 'r') as stream:
        try:
            raw_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError("Bad yaml config: {}".format(exc))
    plot_configs = []
    for raw_config in raw_configs:
        dim_slices = load_dim_slices(raw_config)
        functions, kwargs = load_ufuncs(raw_config)
        plot_config = PlotConfig(
            plot_name=raw_config['plot_name'],
            plot_type=raw_config['plot_type'],
            diagnostic_variable=raw_config['diagnostic_variable'],
            dim_slices=dim_slices,
            functions=functions,
            function_kwargs=kwargs
        )
        plot_configs.append(plot_config)
    return plot_configs



