from dataclasses import dataclass
from typing import List

import numba
import yaml
from vcm.calc import diag_ufuncs

del numba  # imports numba separately to avoid later vcm import crash


@dataclass
class PlotConfig:
    """
    Stores instructions for creating a single figure.
    """

    diagnostic_variable: str
    plot_name: str
    plotting_function: str
    dim_slices: dict
    functions: List
    function_kwargs: List[dict]
    plot_params: dict


def load_ufuncs(raw_config):
    if "function_specifications" in raw_config:
        functions, kwargs = [], []
        for function_spec in raw_config["function_specifications"]:
            # sacrificing some ugliness here so that specification in the yaml is
            # safer / more readable: single entry dict seemed like a more foolproof
            # format than list of func name and kwargs
            function_name, function_kwargs = (
                list(function_spec.keys())[0],
                list(function_spec.values())[0],
            )
            # handle case where function name is given with no kwargs attached
            if not function_kwargs:
                function_kwargs = {}
            if not hasattr(diag_ufuncs, function_name):
                raise ValueError(
                    f"Function name {function_name} is not in the function map."
                )
            functions.append(getattr(diag_ufuncs, function_name))
            kwargs.append(function_kwargs)
        return functions, kwargs
    else:
        return [], []


def load_dim_slices(raw_config):
    dim_selection = {}
    if "dim_slices" in raw_config:
        for dim, indices in raw_config["dim_slices"].items():
            if len(indices) == 1:
                dim_selection[dim] = indices[0]
            else:
                indices += [None]
                dim_selection[dim] = slice(indices[0], indices[1], indices[2])
        return dim_selection
    else:
        return {}


def load_plot_params(raw_config):
    if "plot_params" in raw_config and raw_config["plot_params"] is not None:
        return raw_config["plot_params"]
    else:
        return {}


def load_configs(config_path):
    with open(config_path, "r") as stream:
        try:
            raw_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")
    plot_configs = []
    for raw_config in raw_configs:
        dim_slices = load_dim_slices(raw_config)
        functions, function_kwargs = load_ufuncs(raw_config)
        plot_params = load_plot_params(raw_config)
        plot_config = PlotConfig(
            plot_name=raw_config["plot_name"],
            plotting_function=raw_config["plotting_function"],
            diagnostic_variable=raw_config["diagnostic_variable"],
            dim_slices=dim_slices,
            functions=functions,
            function_kwargs=function_kwargs,
            plot_params=plot_params,
        )
        plot_configs.append(plot_config)
    return plot_configs
